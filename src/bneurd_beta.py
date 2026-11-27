"""Adaptação das rotinas do repositório bneurd/tutoriais para o Dataset BETA (SSVEP).

Este módulo consolida e adapta fielmente todas as etapas propostas nos notebooks
do grupo BNEURD (https://github.com/bneurd/tutoriais) — originalmente demonstradas
no dataset AVI SSVEP (1 canal, 512 Hz, 7 alvos) e em rascunhos com dados fictícios
do BETA — para operar diretamente sobre os arquivos reais do Dataset BETA
(64 canais, 250 Hz, 4 blocos, 40 alvos de 8.0 a 15.8 Hz).

Etapas cobertas:
1. Carregamento e padronização dos arquivos .mat do BETA -> (160, C, T) e metadados;
2. Filtros temporais Butterworth (bandpass, lowpass, highpass, notch 50 Hz);
3. Construção de objetos MNE (mne.EpochsArray) e recorte de janelas (ruído vs. estímulo);
4. Classificação não supervisionada por isolamento de frequência (PSD / FFT + voto);
5. Extração de características reais: narrow-band SNR, wide-band SNR, amplitudes e
   médias espectrais nos 40 alvos + métricas temporais/espectrais (VAR, RMS, FMD, MMDF);
6. Seleção de características (VarianceThreshold, SelectKBest, SelectPercentile, RFE)
   e classificação supervisionada com SVM + validação cruzada por blocos e tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import scipy.io
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Taxa de amostragem e janelas temporais oficiais do BETA (S1-S15: 3.0s = 750 amostras)
BETA_SFREQ = 250
BETA_SORTED_FREQS = np.round(np.arange(8.0, 16.0, 0.2), 1)  # 40 alvos: 8.0 a 15.8 Hz

# Índices (0-based) verificados diretamente em data.suppl_info.chan dos arquivos S*.mat:
# 47: PZ, 53: PO5, 54: PO3, 55: POZ, 56: PO4, 57: PO6, 60: O1, 61: OZ, 62: O2
CHANNEL_PRESETS: Dict[str, Optional[List[int]]] = {
    "all": None,  # Todos os 64 canais
    "9ch": [47, 53, 54, 55, 56, 57, 60, 61, 62],  # 9 parieto-occipitais padrão
    "5ch": [55, 54, 56, 61, 60],                  # 5 canais reduzidos (POZ, PO3, PO4, OZ, O1)
    "3ch": [60, 61, 62],                          # 3 canais mínimos (O1, OZ, O2)
    "oz": [61],                                   # 1 canal central (OZ, análogo ao AVI)
}

# Cortes temporais em segundos dentro do ensaio de 3.0s (para S1-S15):
# [0.0, 0.5)s -> pré-estímulo (ruído basal)
# [0.5, 2.5)s -> janela de estímulo visual (com 130 ms de latência visual inicial)
# [0.63, 2.5)s -> janela de estímulo estacionário compensando 130 ms de latência
# [2.5, 3.0)s -> pós-estímulo / descanso (ruído)
TIME_WINDOWS_SEC: Dict[str, Tuple[float, float]] = {
    "full": (0.0, 3.0),
    "baseline": (0.0, 0.5),
    "stim": (0.5, 2.5),
    "stim_latency": (0.63, 2.5),
    "stim_1s": (0.63, 1.63),
    "rest": (2.5, 3.0),
}


# =============================================================================
# 1. Carregamento e Padronização (Adaptação de load_single_avi.ipynb)
# =============================================================================

def resolve_channel_indices(
    channels: Union[str, Sequence[int], None] = "all",
    total_channels: int = 64,
) -> List[int]:
    """Converte um nome de preset ('all', '9ch', '5ch', '3ch', 'oz') ou lista de índices."""
    if channels is None or channels == "all":
        return list(range(total_channels))
    if isinstance(channels, str):
        key = channels.lower()
        if key not in CHANNEL_PRESETS:
            raise ValueError(f"Preset de canais desconhecido: {channels}. Use {list(CHANNEL_PRESETS)}.")
        preset = CHANNEL_PRESETS[key]
        return list(range(total_channels)) if preset is None else list(preset)
    return [int(idx) for idx in channels]


def load_beta_subject(
    subject_id: int = 1,
    channels: Union[str, Sequence[int], None] = "all",
    window: Union[str, Tuple[float, float]] = "full",
    raw_dir: Path = RAW_DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Carrega um participante do dataset BETA e organiza no padrão (n_epochs, n_channels, n_times).

    Retorna:
        X: ndarray float32 de shape (160, n_channels, n_times) em microvolts (uV).
        y_freqs: ndarray float64 de shape (160,) com a frequência estimulada (Hz) de cada época.
        y_class: ndarray int64 de shape (160,) com o índice de classe ordenado (0=8.0Hz .. 39=15.8Hz).
        blocks: ndarray int64 de shape (160,) indicando o bloco experimental (0, 1, 2 ou 3).
        meta: dicionário com metadados (ch_names, sfreq, raw_freqs, phases, age, gender, bci_quotient).
    """
    mat_path = Path(raw_dir) / f"S{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Arquivo {mat_path} não encontrado. Execute scripts/download_beta.py.")

    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data_obj = mat["data"]
    eeg_raw = np.asarray(data_obj.EEG, dtype=np.float32)  # (64, T_total, 4, 40)
    suppl = data_obj.suppl_info

    sfreq = int(getattr(suppl, "srate", BETA_SFREQ))
    all_ch_names = [str(row[3]).upper() for row in suppl.chan]
    raw_freqs = np.round(np.asarray(suppl.freqs, dtype=np.float64), 1)
    raw_phases = np.asarray(suppl.phases, dtype=np.float64)

    ch_idx = resolve_channel_indices(channels, total_channels=eeg_raw.shape[0])
    ch_names = [all_ch_names[i] for i in ch_idx]

    if isinstance(window, str):
        if window not in TIME_WINDOWS_SEC:
            raise ValueError(f"Janela desconhecida: {window}. Opções: {list(TIME_WINDOWS_SEC)}")
        t_start, t_end = TIME_WINDOWS_SEC[window]
    else:
        t_start, t_end = float(window[0]), float(window[1])

    s_start = int(round(t_start * sfreq))
    s_end = int(round(t_end * sfreq))
    eeg_sliced = eeg_raw[ch_idx, s_start:s_end, :, :]  # (C, T, 4, 40)

    # Ordena os 40 alvos em ordem crescente de frequência (8.0, 8.2, ..., 15.8 Hz)
    freq_order = np.argsort(raw_freqs)
    eeg_sorted = eeg_sliced[:, :, :, freq_order]  # (C, T, 4, 40)
    sorted_freqs = raw_freqs[freq_order]
    sorted_phases = raw_phases[freq_order]

    # Reorganiza para (bloco, alvo, canal, tempo) -> (160, C, T)
    # Assim cada bloco de 40 amostras contém todas as 40 classes sem embaralhar blocos!
    eeg_btct = np.transpose(eeg_sorted, (2, 3, 0, 1))  # (4, 40, C, T)
    n_blocks, n_targets, n_ch, n_times = eeg_btct.shape
    X = eeg_btct.reshape(n_blocks * n_targets, n_ch, n_times).astype(np.float32)

    y_class = np.tile(np.arange(n_targets, dtype=np.int64), n_blocks)
    y_freqs = np.tile(sorted_freqs, n_blocks)
    blocks = np.repeat(np.arange(n_blocks, dtype=np.int64), n_targets)

    meta = {
        "subject_id": f"S{subject_id}",
        "sfreq": sfreq,
        "ch_names": ch_names,
        "ch_indices": ch_idx,
        "all_ch_names": all_ch_names,
        "freqs": sorted_freqs,
        "phases": sorted_phases,
        "window_sec": (t_start, t_end),
        "age": float(getattr(suppl, "age", 0.0)),
        "gender": str(getattr(suppl, "gender", "unknown")),
        "bci_quotient": float(getattr(suppl, "bci_quotient", 0.0)),
        "wide_snr_db": float(np.mean(getattr(suppl, "wide_snr", 0.0))),
    }
    return X, y_freqs, y_class, blocks, meta


# =============================================================================
# 2. Filtros Digitais Temporais (Adaptação de SSVEP/preprocessing.ipynb)
# =============================================================================

def butter_bandpass(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = BETA_SFREQ,
    order: int = 4,
) -> np.ndarray:
    """Filtro Butterworth passa-faixa de fase zero (aplicado no último eixo temporal)."""
    nyq = 0.5 * fs
    sos = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def butter_lowpass(
    data: np.ndarray,
    highcut: float,
    fs: float = BETA_SFREQ,
    order: int = 4,
) -> np.ndarray:
    """Filtro Butterworth passa-baixa de fase zero."""
    nyq = 0.5 * fs
    sos = signal.butter(order, highcut / nyq, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def butter_highpass(
    data: np.ndarray,
    lowcut: float,
    fs: float = BETA_SFREQ,
    order: int = 4,
) -> np.ndarray:
    """Filtro Butterworth passa-alta de fase zero."""
    nyq = 0.5 * fs
    sos = signal.butter(order, lowcut / nyq, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def butter_notch(
    data: np.ndarray,
    cutoff: float = 50.0,
    var: float = 1.0,
    fs: float = BETA_SFREQ,
    order: int = 4,
) -> np.ndarray:
    """Filtro corta-faixa (notch) Butterworth de fase zero (50 Hz no dataset BETA)."""
    nyq = 0.5 * fs
    low = (cutoff - var) / nyq
    high = (cutoff + var) / nyq
    sos = signal.iirfilter(order, [low, high], btype="bandstop", ftype="butter", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


# =============================================================================
# 3. Criação de Objetos MNE (Adaptação de preprocessing.ipynb e beta_ml.ipynb)
# =============================================================================

def create_beta_epochs(
    X_uV: np.ndarray,
    y_freqs: np.ndarray,
    ch_names: List[str],
    sfreq: float = BETA_SFREQ,
    tmin: float = 0.0,
) -> mne.EpochsArray:
    """Converte o array (n_epochs, n_channels, n_times) em uV para mne.EpochsArray (em Volts)."""
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    unique_freqs = np.unique(np.round(y_freqs, 1))
    event_id = {f"{freq:.1f}": idx for idx, freq in enumerate(unique_freqs)}
    freq_to_code = {round(float(freq), 1): idx for idx, freq in enumerate(unique_freqs)}

    events = np.column_stack([
        np.arange(len(y_freqs), dtype=int),
        np.zeros(len(y_freqs), dtype=int),
        np.array([freq_to_code[round(float(f), 1)] for f in y_freqs], dtype=int),
    ])

    # MNE armazena EEG em Volts (1 uV = 1e-6 V)
    X_volts = np.asarray(X_uV, dtype=np.float64) * 1e-6
    epochs = mne.EpochsArray(X_volts, info=info, events=events, tmin=tmin, event_id=event_id, verbose=False)
    return epochs


# =============================================================================
# 4. Classificador por Isolamento de Frequência (classifier_per_freq.ipynb)
# =============================================================================

def classify_by_psd_voting(
    epochs: mne.EpochsArray,
    target_freqs: Sequence[float] = BETA_SORTED_FREQS,
    threshold: float = 0.08,
    agg_mode: str = "max",
    method: str = "periodogram",
    n_harmonics: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Isola as bandas [f - threshold, f + threshold] e classifica pelo voto de maior energia.

    No tutorial original (AVI dataset), havia apenas 7 alvos espaçados por >= 0.5 Hz, usando
    threshold=0.25 Hz. No BETA, os 40 alvos distam apenas 0.2 Hz entre si! Por isso:
    - Usamos threshold < 0.1 Hz (ex: 0.08 Hz) para não invadir a frequência vizinha;
    - Oferecemos tanto `mne.Epochs.compute_psd(method='multitaper')` (fiel ao tutorial)
      quanto FFT com zero-padding (`n_fft=2500`, resolução 0.1 Hz) e soma harmônica opcional.
    """
    X = epochs.get_data(copy=False)  # (n_epochs, n_channels, n_times)
    sfreq = float(epochs.info["sfreq"])
    targets = np.asarray(target_freqs, dtype=np.float64)

    if method == "mne_multitaper":
        # Executa compute_psd nativo do MNE (como em classifier_per_freq.ipynb) com largura de banda padrão
        fmin_all = float(np.min(targets) - threshold - 0.5)
        fmax_all = float(min(sfreq / 2.0 - 1.0, np.max(targets) * n_harmonics + threshold + 0.5))
        psd_obj = epochs.compute_psd(method="multitaper", fmin=fmin_all, fmax=fmax_all, verbose=False)
        psds, freqs = psd_obj.get_data(return_freqs=True)  # (n_epochs, n_ch, n_freqs)
    else:
        # Periodograma FFT com zero-padding para resolução fina de 0.05 Hz (n_fft=5000)
        n_fft = 5000
        fft_vals = np.fft.rfft(X, n=n_fft, axis=-1)
        psds = (np.abs(fft_vals) ** 2) / X.shape[-1]
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

    # Z-score espectral por canal ou média espacial nos canais selecionados
    scores = np.zeros((X.shape[0], len(targets)), dtype=np.float64)
    for t_idx, f0 in enumerate(targets):
        harmonic_score = np.zeros(X.shape[0], dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            fh = h * f0
            mask = (freqs >= fh - threshold) & (freqs <= fh + threshold)
            if not np.any(mask):
                nearest = np.argmin(np.abs(freqs - fh))
                band_vals = psds[:, :, nearest : nearest + 1]
            else:
                band_vals = psds[:, :, mask]  # (n_epochs, n_ch, n_bins)

            if agg_mode == "max":
                feat_ch = np.max(band_vals, axis=-1)
            elif agg_mode == "mean":
                feat_ch = np.mean(band_vals, axis=-1)
            elif agg_mode == "median":
                feat_ch = np.median(band_vals, axis=-1)
            else:
                raise ValueError(f"agg_mode inválido: {agg_mode}")

            # Peso harmônico decrescente (1/h) e média entre os canais selecionados
            harmonic_score += (1.0 / h) * np.mean(feat_ch, axis=-1)
        scores[:, t_idx] = harmonic_score

    y_pred_idx = np.argmax(scores, axis=-1)
    y_pred_freqs = targets[y_pred_idx]
    return y_pred_freqs, scores


# =============================================================================
# 5. Extração de Características SNR, FFT e Temporais (beta_ml.ipynb + tunning.ipynb)
# =============================================================================

def compute_background_noise_power(
    X_full: np.ndarray,
    sfreq: float = BETA_SFREQ,
    n_fft: int = 1250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estima o espectro de potência do ruído de fundo usando os trechos sem estímulo do BETA.

    Conforme descrito em `beta_ml.ipynb`:
    - Pré-estímulo: 0.0 a 0.5 s (amostras 0 a 125)
    - Pós-estímulo (descanso): 2.5 a 3.0 s (amostras 625 a 750)

    Retorna:
        noise_psd_bins: shape (n_epochs, n_channels, n_freqs) - média do PSD basal e descanso.
        mean_noise_per_channel: shape (n_epochs, n_channels) - potência média escalar de ruído.
    """
    base_seg = X_full[:, :, 0:125]
    rest_seg = X_full[:, :, 625:750]

    # Normaliza pelo número de amostras do segmento antes do zero-padding
    psd_base = (np.abs(np.fft.rfft(base_seg, n=n_fft, axis=-1)) ** 2) / base_seg.shape[-1]
    psd_rest = (np.abs(np.fft.rfft(rest_seg, n=n_fft, axis=-1)) ** 2) / rest_seg.shape[-1]

    noise_psd_bins = 0.5 * (psd_base + psd_rest)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    band_mask = (freqs >= 8.0) & (freqs <= 64.0)
    mean_noise_per_channel = np.mean(noise_psd_bins[:, :, band_mask], axis=-1)
    return noise_psd_bins, mean_noise_per_channel


def extract_beta_ml_features(
    X_full: np.ndarray,
    target_freqs: Sequence[float] = BETA_SORTED_FREQS,
    sfreq: float = BETA_SFREQ,
    stim_samples: Tuple[int, int] = (157, 625),
    n_fft: int = 1250,
    neighbor_bins: int = 5,
    n_harmonics: int = 3,
) -> Dict[str, np.ndarray]:
    """Extrai as 4 famílias de características propostas em `beta_ml.ipynb` e `tunning.ipynb`.

    Usa dados reais do BETA (resolvendo os problemas de valores negativos/NaN do rascunho fictício):
    1. `narrow_snr` (160, C, 40): SNR de banda estreita em dB em cada alvo (potência no alvo
       dividida pela média das frequências vizinhas, Eq. 3 do artigo BETA).
    2. `wide_snr` (160, C, 40): SNR de banda larga em dB em cada alvo (soma harmônica dividida
       pela potência total do espectro 8-64 Hz, Eq. 4 do artigo BETA).
    3. `basal_snr` (160, C, 40): Razão em dB entre a potência na janela estimulada e a potência
       do ruído de fundo medida em [0.0-0.5s] e [2.5-3.0s] (proposta de `beta_ml.ipynb`).
    4. `target_amp` (160, C, 40): Amplitude espectral combinada (fundamental + harmônicos).
    5. `feature_matrix` (160, C * 40 * 4): Matriz achatada pronta para Seleção de Características e SVM.
    """
    eps = 1e-12
    targets = np.asarray(target_freqs, dtype=np.float64)
    stim_seg = X_full[:, :, stim_samples[0] : stim_samples[1]]

    # FFT com n_fft=1250 -> resolução exata de 250 / 1250 = 0.2 Hz por bin!
    fft_stim = np.fft.rfft(stim_seg, n=n_fft, axis=-1)
    psd_stim = (np.abs(fft_stim) ** 2) / stim_seg.shape[-1]  # (N, C, F)
    amp_stim = np.sqrt(psd_stim + eps)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

    noise_psd_bins, _ = compute_background_noise_power(X_full, sfreq=sfreq, n_fft=n_fft)

    band_8_64 = (freqs >= 8.0) & (freqs <= 64.0)
    total_band_power = np.sum(psd_stim[:, :, band_8_64], axis=-1, keepdims=True) + eps  # (N, C, 1)

    n_epochs, n_ch, _ = psd_stim.shape
    n_targets = len(targets)

    narrow_snr = np.zeros((n_epochs, n_ch, n_targets), dtype=np.float32)
    wide_snr = np.zeros((n_epochs, n_ch, n_targets), dtype=np.float32)
    basal_snr = np.zeros((n_epochs, n_ch, n_targets), dtype=np.float32)
    target_amp = np.zeros((n_epochs, n_ch, n_targets), dtype=np.float32)

    for k, f0 in enumerate(targets):
        idx0 = int(np.argmin(np.abs(freqs - f0)))

        # Vizinhança para narrow-band SNR (exclui o bin central idx0)
        left = max(1, idx0 - neighbor_bins)
        right = min(psd_stim.shape[-1] - 1, idx0 + neighbor_bins)
        neigh_indices = [b for b in range(left, right + 1) if b != idx0]
        neigh_power = np.mean(psd_stim[:, :, neigh_indices], axis=-1) + eps

        # Soma harmônica ponderada (h = 1..n_harmonics)
        harm_power = np.zeros((n_epochs, n_ch), dtype=np.float64)
        harm_noise = np.zeros((n_epochs, n_ch), dtype=np.float64)
        harm_amp = np.zeros((n_epochs, n_ch), dtype=np.float64)

        for h in range(1, n_harmonics + 1):
            idx_h = int(np.argmin(np.abs(freqs - h * f0)))
            harm_power += psd_stim[:, :, idx_h]
            harm_noise += noise_psd_bins[:, :, idx_h]
            harm_amp += (1.0 / h) * amp_stim[:, :, idx_h]

        p_fund = psd_stim[:, :, idx0] + eps
        narrow_snr[:, :, k] = 10.0 * np.log10(p_fund / neigh_power)

        denom_wide = np.maximum(total_band_power[:, :, 0] - harm_power, eps)
        wide_snr[:, :, k] = 10.0 * np.log10((harm_power + eps) / denom_wide)

        basal_snr[:, :, k] = 10.0 * np.log10((harm_power + eps) / (harm_noise + eps))
        target_amp[:, :, k] = harm_amp.astype(np.float32)

    # Concatena as 4 matrizes (160, C, 40) -> (160, C, 160) -> (160, C * 160) conforme tunning.ipynb!
    stacked = np.concatenate([narrow_snr, wide_snr, basal_snr, target_amp], axis=-1)
    feature_matrix = stacked.reshape(n_epochs, -1).astype(np.float32)

    return {
        "narrow_snr": narrow_snr,
        "wide_snr": wide_snr,
        "basal_snr": basal_snr,
        "target_amp": target_amp,
        "feature_matrix": feature_matrix,
        "freqs_fft": freqs,
        "psd_stim": psd_stim,
        "psd_noise": noise_psd_bins,
    }


def extract_classic_channel_features(X_epoch: np.ndarray) -> np.ndarray:
    """Extrai as características clássicas de `SomeFeatures/04-features.ipynb` (VAR, RMS, FMD, MMDF).

    Entrada:
        X_epoch: (n_epochs, n_channels, n_times)
    Saída:
        X_feat: (n_epochs, n_channels * 4)
    """
    n_times = X_epoch.shape[-1]
    X_freq = np.fft.rfft(X_epoch, axis=-1)
    psd = (np.abs(X_freq) ** 2) / n_times

    feat_var = np.sum(X_epoch ** 2, axis=-1) / max(n_times - 1, 1)
    feat_rms = np.sqrt(np.mean(X_epoch ** 2, axis=-1))
    feat_fmd = 0.5 * np.sum(psd, axis=-1)
    feat_mmdf = 0.5 * np.sum(np.abs(X_freq), axis=-1)

    stacked = np.stack([feat_var, feat_rms, feat_fmd, feat_mmdf], axis=-1)  # (N, C, 4)
    return stacked.reshape(X_epoch.shape[0], -1).astype(np.float32)


# =============================================================================
# 6. Seleção de Características e Classificação SVM (tunning.ipynb)
# =============================================================================

def evaluate_selectors_and_svm(
    X_feat: np.ndarray,
    y: np.ndarray,
    blocks: Optional[np.ndarray] = None,
    k_best: int = 160,
    percentile: int = 25,
    rfe_features: int = 160,
    rfe_step: float = 0.25,
) -> Dict[str, float]:
    """Compara os seletores de características do tutorial seguidos por classificador SVM.

    Usa validação cruzada Leave-One-Block-Out (4 folds correspondentes aos 4 blocos do BETA)
    dentro de um Pipeline sklearn (sem vazamento de dados entre treino e teste!).
    """
    if blocks is not None:
        cv = LeaveOneGroupOut()
        groups = blocks
    else:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        groups = None

    n_total = X_feat.shape[1]
    k_eff = min(k_best, n_total)
    rfe_eff = min(rfe_features, n_total)

    pipelines = {
        "SVM Puro (Sem Seleção)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=0.1)),
        ]),
        "VarianceThreshold + SVM": Pipeline([
            ("var", VarianceThreshold(threshold=1e-4)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=0.1)),
        ]),
        f"SelectKBest (k={k_eff}) + SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(score_func=f_classif, k=k_eff)),
            ("svm", SVC(kernel="linear", C=0.1)),
        ]),
        f"SelectPercentile ({percentile}%) + SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("perc", SelectPercentile(score_func=f_classif, percentile=percentile)),
            ("svm", SVC(kernel="linear", C=0.1)),
        ]),
    }

    # RFE é computacionalmente mais caro quando há > 2000 colunas; pré-filtramos as top-400 por ANOVA
    if n_total > 400:
        pipelines[f"RFE (n={rfe_eff}) + SVM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pre_kbest", SelectKBest(score_func=f_classif, k=400)),
            ("rfe", RFE(estimator=SVC(kernel="linear", C=0.1), n_features_to_select=min(rfe_eff, 400), step=rfe_step)),
            ("svm", SVC(kernel="linear", C=0.1)),
        ])
    else:
        pipelines[f"RFE (n={rfe_eff}) + SVM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("rfe", RFE(estimator=SVC(kernel="linear", C=0.1), n_features_to_select=rfe_eff, step=rfe_step)),
            ("svm", SVC(kernel="linear", C=0.1)),
        ])

    results: Dict[str, float] = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_feat, y, groups=groups, cv=cv, n_jobs=1)
        results[name] = float(np.mean(scores) * 100.0)
    return results


def tune_svm_hyperparameters(
    X_feat: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    k_features: int = 200,
) -> Tuple[Dict, float, List[Dict]]:
    """Realiza o tuning de hiperparâmetros do SVM (kernel, C, gamma) conforme `tunning.ipynb`."""
    k_eff = min(k_features, X_feat.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_eff)),
        ("svm", SVC()),
    ])

    param_grid = [
        {"select__k": [min(120, k_eff), k_eff], "svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1.0]},
        {"select__k": [min(120, k_eff), k_eff], "svm__kernel": ["rbf"], "svm__C": [1.0, 10.0], "svm__gamma": ["scale", 1e-3]},
    ]

    cv = LeaveOneGroupOut()
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=1)
    grid.fit(X_feat, y, groups=blocks)

    history = []
    for params, mean_score, std_score in zip(
        grid.cv_results_["params"],
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
    ):
        history.append({
            "params": params,
            "mean_acc_pct": float(mean_score * 100.0),
            "std_acc_pct": float(std_score * 100.0),
        })

    return grid.best_params_, float(grid.best_score_ * 100.0), history
