"""Constrói e executa os 5 Jupyter Notebooks adaptados de `bneurd/tutoriais` para a base BETA.

Gera arquivos `.ipynb` (nbformat v4.5) em `notebooks/tutoriais_beta/`, executando cada
célula de código sobre os dados reais (`data/raw/S1.mat..S10.mat`) e embutindo saídas de
texto e figuras PNG diretamente nos notebooks para inspeção imediata no VS Code e GitHub.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOKS_OUT_DIR = PROJECT_ROOT / "notebooks" / "tutoriais_beta"
SUMMARY_JSON_PATH = PROJECT_ROOT / "experiments" / "results" / "tutoriais_beta_summary.json"


def md_cell(text: str) -> Dict[str, Any]:
    lines = text.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str) -> Dict[str, Any]:
    lines = code.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def execute_notebook_cells(cells: List[Dict[str, Any]], nb_name: str) -> Dict[str, Any]:
    """Executa sequencialmente todas as células de código, capturando stdout, repr e figuras PNG."""
    ns: Dict[str, Any] = {"__name__": "__main__"}
    exec_counter = 1

    for idx, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue

        raw_src = "".join(cell["source"])
        # Ignora diretivas mágicas IPython como %matplotlib inline
        clean_lines = [ln for ln in raw_src.splitlines() if not ln.strip().startswith("%")]
        code_str = "\n".join(clean_lines)

        stdout_buf = io.StringIO()
        outputs: List[Dict[str, Any]] = []
        captured_figs: List[bytes] = []

        plt.close("all")

        def _capture_active_figs() -> None:
            for num in list(plt.get_fignums()):
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                captured_figs.append(buf.getvalue())
            plt.close("all")

        orig_show = plt.show
        plt.show = lambda *a, **kw: _capture_active_figs()

        try:
            import ast
            tree = ast.parse(code_str, filename=f"{nb_name}_cell_{idx}")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr_node = tree.body.pop()
                last_expr = ast.Expression(last_expr_node.value)

            with contextlib.redirect_stdout(stdout_buf):
                if tree.body:
                    exec(compile(tree, filename=f"{nb_name}_cell_{idx}", mode="exec"), ns)
                result_val = None
                if last_expr is not None:
                    result_val = eval(compile(last_expr, filename=f"{nb_name}_cell_{idx}", mode="eval"), ns)

            # Captura figuras remanescentes mesmo que plt.show() não tenha sido chamado
            if plt.get_fignums():
                _capture_active_figs()
        except Exception as exc:
            plt.show = orig_show
            tb = traceback.format_exc()
            raise RuntimeError(f"Erro ao executar {nb_name} (célula {idx}):\n{code_str}\n\n{tb}") from exc
        finally:
            plt.show = orig_show

        out_text = stdout_buf.getvalue()
        if out_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": out_text.splitlines(keepends=True),
            })

        for png_bytes in captured_figs:
            b64_png = base64.b64encode(png_bytes).decode("ascii")
            outputs.append({
                "data": {
                    "image/png": b64_png,
                    "text/plain": ["<Figure>"],
                },
                "metadata": {},
                "output_type": "display_data",
            })

        if result_val is not None:
            outputs.append({
                "data": {
                    "text/plain": repr(result_val).splitlines(keepends=True),
                },
                "execution_count": exec_counter,
                "metadata": {},
                "output_type": "execute_result",
            })

        cell["execution_count"] = exec_counter
        cell["outputs"] = outputs
        exec_counter += 1

    return ns


def save_notebook(cells: List[Dict[str, Any]], out_path: Path) -> None:
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb_dict, ensure_ascii=False, indent=1), encoding="utf-8")


COMMON_HEADER_CODE = """\
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
"""


def build_nb01() -> List[Dict[str, Any]]:
    return [
        md_cell("""
# 01 — Introdução, Protocolo e Carregamento da Base BETA

> **Adaptação de:** `SSVEP/intro.ipynb`, `SSVEP/load_single_avi.ipynb` e `SSVEP/load_multi_avi_exemplo.ipynb` ([bneurd/tutoriais](https://github.com/bneurd/tutoriais)) para o **Dataset BETA** (Liu et al., 2020).

## Comparativo entre o Dataset Original do Tutorial (AVI) e a Nossa Base (BETA)

| Característica | AVI SSVEP (Tutorial Original) | BETA Dataset (Nossa Adaptação) |
| :--- | :--- | :--- |
| **Participantes** | 4 (*Single-Target*) / 5 (*Multi-Target*) | **70 voluntários** (`S1` a `S10` locais para prototipação) |
| **Eletrodos Gravados** | 1 canal (`Oz`) | **64 canais** (padrão 10-20; foco nos 9 parieto-occipitais) |
| **Taxa de Amostragem** | 512 Hz | **250 Hz** |
| **Alvos / Frequências** | 7 frequências (6.0 a 10.0 Hz) | **40 alvos** (8.0 a 15.8 Hz, passo de 0.2 Hz, 4 fases) |
| **Estrutura por Sujeito** | 1–2 sessões longas (16s a 30s) | **4 blocos × 40 alvos = 160 ensaios** de 3.0s (`S1–S15`) |
"""),
        code_cell(COMMON_HEADER_CODE + """
from src.bneurd_beta import load_beta_subject, CHANNEL_PRESETS, PROCESSED_DATA_DIR

print("Diretório raiz do projeto:", PROJECT_ROOT)
print("Presets espaciais configurados:", CHANNEL_PRESETS)
"""),
        md_cell("""
### 1. Inspeção de todos os 10 participantes locais (`S1.mat` a `S10.mat`)
Assim como em `load_single_avi.ipynb` e `load_multi_avi_exemplo.ipynb`, inspecionamos a divisão dos dados, os rótulos de frequência e os metadados fisiológicos de cada participante.
"""),
        code_cell("""
subjects_meta = []
for sub_id in range(1, 11):
    X_sub, y_freqs_sub, y_cls_sub, blocks_sub, meta = load_beta_subject(sub_id, channels="all", window="full")
    subjects_meta.append(meta)
    print(
        f"{meta['subject_id']:>3s} | shape={X_sub.shape} | "
        f"idade={meta['age']:4.1f} | gênero={meta['gender']:6s} | "
        f"BCI_Q={meta['bci_quotient']:5.1f} | Wide_SNR={meta['wide_snr_db']:6.2f} dB"
    )

print("\\nSequência ordenada das 40 frequências alvo (Hz):")
print(subjects_meta[0]["freqs"])
"""),
        md_cell("""
### 2. Padronização e Salvamento em `.npy`
No tutorial original (`load_single_avi.ipynb`), os ensaios são padronizados e salvos em arquivos `.npy` (`data.npy` e `labels.npy`) para consumo nos notebooks seguintes. Aqui salvamos o participante `S1` padronizado no formato `(160, 64, 750)` (4 blocos × 40 alvos = 160 tentativas):
"""),
        code_cell("""
X_s1, y_freqs_s1, y_cls_s1, blocks_s1, meta_s1 = load_beta_subject(1, channels="all", window="full")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
np.save(PROCESSED_DATA_DIR / "beta_s1_data.npy", X_s1)
np.save(PROCESSED_DATA_DIR / "beta_s1_labels_freq.npy", y_freqs_s1)
np.save(PROCESSED_DATA_DIR / "beta_s1_blocks.npy", blocks_s1)

print("Arquivos salvos em data/processed/:")
print(" - beta_s1_data.npy       :", X_s1.shape, X_s1.dtype)
print(" - beta_s1_labels_freq.npy:", y_freqs_s1.shape, y_freqs_s1[:10])
print(" - 9 canais occipitais    :", [meta_s1["all_ch_names"][i] for i in CHANNEL_PRESETS["9ch"]])
"""),
        md_cell("""
### 3. Visualização Exploratória Inicial (Perfil dos Participantes e Traçado Bruto no Canal `OZ`)
"""),
        code_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

sub_names = [m["subject_id"] for m in subjects_meta]
bci_q = [m["bci_quotient"] for m in subjects_meta]
snrs = [m["wide_snr_db"] for m in subjects_meta]

axes[0].bar(sub_names, bci_q, color="#2b6cb0", alpha=0.85)
axes[0].axhline(100, color="crimson", linestyle="--", linewidth=1, label="Média Populacional (100)")
axes[0].set_title("Quociente BCI por Voluntário (S1 a S10)")
axes[0].set_ylabel("BCI Quotient")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

oz_idx = CHANNEL_PRESETS["oz"][0]
times = np.arange(X_s1.shape[-1]) / meta_s1["sfreq"]
trial_10hz = np.where(np.isclose(y_freqs_s1, 10.0))[0][0]

axes[1].plot(times, X_s1[trial_10hz, oz_idx, :], color="#1a202c", linewidth=0.9)
axes[1].axvspan(0.0, 0.5, color="#feb2b2", alpha=0.35, label="Pré-estímulo (0.0-0.5s)")
axes[1].axvspan(0.5, 2.5, color="#9ae6b4", alpha=0.30, label="Estímulo SSVEP (0.5-2.5s)")
axes[1].axvspan(2.5, 3.0, color="#fbd38d", alpha=0.35, label="Descanso (2.5-3.0s)")
axes[1].axvline(0.63, color="darkgreen", linestyle="--", label="Fim da latência (0.63s)")
axes[1].set_title(f"Sinal Bruto em OZ — S1 (Alvo: {y_freqs_s1[trial_10hz]:.1f} Hz)")
axes[1].set_xlabel("Tempo (s)")
axes[1].set_ylabel("Amplitude (uV)")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
"""),
    ]


def build_nb02() -> List[Dict[str, Any]]:
    return [
        md_cell("""
# 02 — Visualização, Filtros Temporais e Objetos MNE no Dataset BETA

> **Adaptação de:** `SSVEP/preprocessing.ipynb` e seção MNE de `SSVEP/beta_ml.ipynb` ([bneurd/tutoriais](https://github.com/bneurd/tutoriais)).

Neste notebook executamos exatamente as duas tarefas propostas pelo professor nos tutoriais originais:
1. Implementar e visualizar os filtros temporais Butterworth (**passa-faixa**, **passa-baixa**, **passa-alta** e **notch em 50 Hz**) nos domínios do tempo, PSD e espectrograma;
2. Criar o objeto `mne.EpochsArray` para o **Dataset BETA**, filtrá-lo e recortar as janelas temporais para comprovar visualmente:
   * **Janelas (a) `0.0 – 0.5 s` e `2.5 – 3.0 s`:** apenas ruído espontâneo (sem pico SSVEP);
   * **Janela (b) `0.5 – 2.5 s` (`0.63 – 2.5 s` sem latência):** pico SSVEP nítido cravado na frequência estimulada!
"""),
        code_cell(COMMON_HEADER_CODE + """
import warnings
warnings.filterwarnings("ignore")

import mne
from scipy.fft import rfft, rfftfreq
from src.bneurd_beta import (
    PROCESSED_DATA_DIR,
    butter_bandpass,
    butter_highpass,
    butter_lowpass,
    butter_notch,
    create_beta_epochs,
    load_beta_subject,
)

X_s1_9ch, y_freqs_s1, y_cls_s1, blocks_s1, meta_s1 = load_beta_subject(1, channels="9ch", window="full")
sfreq = meta_s1["sfreq"]
print("Dados S1 (9 canais occipitais):", X_s1_9ch.shape, "| Canais:", meta_s1["ch_names"])
"""),
        md_cell("""
### 1. Aplicação dos Filtros Butterworth (`bandpass`, `notch`) e Gráficos Tempo / PSD / Espectrograma
Adaptado diretamente da função `print_graphs` de `SSVEP/preprocessing.ipynb`, ajustando `Fs=250 Hz` e filtro *notch* para `50 Hz` (frequência da rede elétrica onde o BETA foi coletado).
"""),
        code_cell("""
# Seleciona o canal OZ (índice 7 dentro dos 9 canais: PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2)
oz_local_idx = meta_s1["ch_names"].index("OZ")
oz_raw = X_s1_9ch[:, oz_local_idx, :]  # (160, 750)

# Aplica Notch 50 Hz + Passa-faixa 8-64 Hz (fundamental + harmônicos)
oz_notched = butter_notch(oz_raw, cutoff=50.0, var=1.5, fs=sfreq, order=4)
oz_filtered = butter_bandpass(oz_notched, lowcut=8.0, highcut=64.0, fs=sfreq, order=4)

# Também demonstra lowpass (64 Hz) e highpass (8 Hz) separadamente
oz_lp = butter_lowpass(oz_raw, highcut=64.0, fs=sfreq)
oz_hp = butter_highpass(oz_lp, lowcut=8.0, fs=sfreq)
print("Verificação numérica filtros:", oz_filtered.shape, oz_hp.shape)

def print_beta_graphs(sig_2d, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    t = np.arange(sig_2d.shape[-1]) / sfreq
    for i in range(0, 160, 10):
        axes[0].plot(t, sig_2d[i, :], alpha=0.5, linewidth=0.8)
    axes[0].set_title(f"{title_prefix} — Domínio do Tempo (OZ)")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(alpha=0.3)

    for i in range(0, 160, 10):
        axes[1].psd(sig_2d[i, 157:625], NFFT=512, Fs=sfreq)
    axes[1].set_xlim(0, 70)
    axes[1].set_title(f"{title_prefix} — PSD (8-64 Hz em destaque)")

    axes[2].specgram(sig_2d[20, :], NFFT=128, Fs=sfreq, noverlap=96, cmap="viridis")
    axes[2].set_ylim(0, 70)
    axes[2].set_title(f"{title_prefix} — Espectrograma (Alvo {y_freqs_s1[20]:.1f} Hz)")
    axes[2].set_xlabel("Tempo (s)")
    axes[2].set_ylabel("Frequência (Hz)")

    plt.tight_layout()
    plt.show()

print_beta_graphs(oz_raw, "Sem Filtro Adicional")
print_beta_graphs(oz_filtered, "Filtrado (Notch 50Hz + Bandpass 8-64Hz)")
"""),
        md_cell("""
### 2. Criação do Objeto `mne.EpochsArray` e Análise das Janelas com vs. sem Estímulo
Seguindo o roteiro exato de `SSVEP/beta_ml.ipynb`:
1. Criar o objeto `MNE` a partir dos dados do participante;
2. Aplicar no objeto `MNE` o filtro passa-faixa (`6 – 18 Hz` para isolar as fundamentais de `8.0 – 15.8 Hz`, e salvando também a versão `8 – 64 Hz` em `.fif`);
3. Criar cópias do objeto `MNE` com fatias menores:
   * **(a)** `0.0 – 0.5 s` (pré-estímulo) e `2.5 – 3.0 s` (pós-estímulo): apenas ruído;
   * **(b)** `0.5 – 2.5 s` (estímulo SSVEP);
4. Aplicar `scipy.fft` e plotar no domínio da frequência comprovando a presença do pico SSVEP apenas na janela estimulada!
"""),
        code_cell("""
mne_epochs = create_beta_epochs(X_s1_9ch, y_freqs_s1, ch_names=meta_s1["ch_names"], sfreq=sfreq, tmin=0.0)
print(mne_epochs)

# Versão filtrada 6-18 Hz (conforme roteiro beta_ml.ipynb para visualização da fundamental)
mne_6_18 = mne_epochs.copy().filter(l_freq=6.0, h_freq=18.0, verbose=False)

# Cópias fatiadas nos 3 momentos do ensaio
epo_pre_noise = mne_6_18.copy().crop(tmin=0.0, tmax=0.496)
epo_stim = mne_6_18.copy().crop(tmin=0.5, tmax=2.496)
epo_post_noise = mne_6_18.copy().crop(tmin=2.5, tmax=2.996)

# Compara via scipy.fft para 3 frequências alvo exemplares: 9.0 Hz, 12.0 Hz e 15.0 Hz
demo_targets = ["9.0", "12.0", "15.0"]
fig, axes = plt.subplots(len(demo_targets), 3, figsize=(15, 9), sharey="row")

for row_idx, f_str in enumerate(demo_targets):
    f_val = float(f_str)
    windows = [
        ("Pré-estímulo (0.0 - 0.5s) [Ruído]", epo_pre_noise[f_str]),
        ("Estímulo SSVEP (0.5 - 2.5s) [Sinal]", epo_stim[f_str]),
        ("Pós-estímulo (2.5 - 3.0s) [Ruído]", epo_post_noise[f_str]),
    ]
    for col_idx, (w_title, ep_slice) in enumerate(windows):
        # Média sobre os 4 blocos e os 9 canais occipitais (convertendo de V para uV)
        sig_mean = ep_slice.get_data(copy=False).mean(axis=(0, 1)) * 1e6
        n_fft = 1250
        amps = np.abs(rfft(sig_mean, n=n_fft)) / len(sig_mean)
        freqs = rfftfreq(n_fft, d=1.0 / sfreq)
        mask = (freqs >= 5.0) & (freqs <= 20.0)

        ax = axes[row_idx, col_idx]
        ax.plot(freqs[mask], amps[mask], color="#2b6cb0" if col_idx == 1 else "#718096", linewidth=1.3)
        ax.axvline(f_val, color="green", linestyle="--", linewidth=1.2, label=f"Alvo {f_val} Hz")
        ax.set_title(f"{w_title}\\nAlvo = {f_val} Hz", fontsize=10)
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel("Amplitude FFT (uV)")
        if row_idx == len(demo_targets) - 1:
            ax.set_xlabel("Frequência (Hz)")
        ax.legend(loc="upper right", fontsize=8)

plt.suptitle("Comparação Espectral (scipy.fft): Janelas sem Estímulo vs. Janela com SSVEP (S1)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""),
        md_cell("""
### 3. Visualização `compute_psd(method='multitaper')` e Salvamento em `.fif`
Como em `SSVEP/preprocessing.ipynb`, verificamos também o método nativo `compute_psd(method='multitaper')` do MNE na banda completa `8–64 Hz` (que preserva os harmônicos $2f$ e $3f$) e salvamos o arquivo `.fif` filtrado em `data/processed/beta_s1_filtered-epo.fif` para o próximo notebook!
"""),
        code_cell("""
mne_filtered_wide = mne_epochs.copy().filter(l_freq=8.0, h_freq=64.0, verbose=False)
mne_stim_clean = mne_filtered_wide.copy().crop(tmin=0.63, tmax=2.496)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
fif_path = PROCESSED_DATA_DIR / "beta_s1_filtered-epo.fif"
mne_stim_clean.save(fif_path, overwrite=True, verbose=False)
print("Arquivo MNE salvo com sucesso em:", fif_path)

# Plota PSD multitaper para 4 alvos demonstrando fundamental + 2º harmônico
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, f_str in zip(axes.flatten(), ["8.6", "10.4", "12.8", "15.2"]):
    f0 = float(f_str)
    psd_spec = mne_stim_clean[f_str].compute_psd(method="multitaper", fmin=6.0, fmax=48.0, verbose=False)
    psds, f_axis = psd_spec.get_data(return_freqs=True)
    psd_db = 10 * np.log10(psds.mean(axis=(0, 1)) + 1e-20)
    ax.plot(f_axis, psd_db, color="#1a365d", linewidth=1.2)
    ax.axvline(f0, color="green", linestyle="--", label=f"1f = {f0:.1f} Hz")
    ax.axvline(2 * f0, color="darkorange", linestyle=":", label=f"2f = {2*f0:.1f} Hz")
    ax.set_title(f"MNE Multitaper PSD — Estímulo {f_str} Hz")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("PSD (dB V²/Hz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
"""),
    ]


def build_nb03() -> List[Dict[str, Any]]:
    return [
        md_cell("""
# 03 — Classificação por Isolamento de Frequência e Voto no BETA

> **Adaptação de:** `SSVEP/classifier_per_freq.ipynb` ([bneurd/tutoriais](https://github.com/bneurd/tutoriais)).

## O que muda do `AVI` (7 classes) para o `BETA` (40 classes)?
1. **Limiar (`threshold`):** No dataset AVI, as 7 frequências (`6.0, 6.5, 7.0, 7.5, 8.2, 9.3, 10.0 Hz`) tinham distância mínima de `0.5 Hz`, permitindo `threshold = 0.25 Hz`. No **BETA**, existem **40 frequências** espaçadas a cada **`0.2 Hz`** (`8.0, 8.2, ..., 15.8 Hz`). Logo, usamos `threshold = 0.08 Hz` para isolar cada raia sem invadir a vizinha!
2. **Múltiplos Canais e Harmônicos:** Testamos exatamente os 3 extratores manuais sugeridos no tutorial (`max`, `mean`, `median`), avaliando o ganho de sair de 1 único canal (`OZ`, como no AVI) para **3, 5 e 9 canais occipitais** e somando **harmônicos ($1f + 2f + 3f$)**!
"""),
        code_cell(COMMON_HEADER_CODE + """
import mne
from src.bneurd_beta import (
    BETA_SORTED_FREQS,
    PROCESSED_DATA_DIR,
    butter_bandpass,
    butter_notch,
    classify_by_psd_voting,
    create_beta_epochs,
    load_beta_subject,
)

fif_path = PROCESSED_DATA_DIR / "beta_s1_filtered-epo.fif"
epochs_s1 = mne.read_epochs(fif_path, verbose=False)
y_true_s1 = np.array([ BETA_SORTED_FREQS[e] for e in epochs_s1.events[:, 2] ])
print("Épocas carregadas de beta_s1_filtered-epo.fif:", epochs_s1.get_data(copy=False).shape)
"""),
        md_cell("""
### 1. Comparação entre `max_value`, `average` e `median` (Multitaper MNE vs. Periodograma FFT) em `S1`
Executamos os 5 passos do roteiro `classifier_per_freq.ipynb` para o participante `S1` (acurácia de chance aleatória em 40 alvos = $1/40 = 2,5\\%$):
"""),
        code_cell("""
for method_name in ["mne_multitaper", "periodogram"]:
    print(f"=== Método Espectral: {method_name} (9 canais, apenas fundamental 1f) ===")
    for agg in ["max", "mean", "median"]:
        y_pred, scores = classify_by_psd_voting(
            epochs_s1, target_freqs=BETA_SORTED_FREQS, threshold=0.08,
            agg_mode=agg, method=method_name, n_harmonics=1
        )
        acc = 100.0 * np.mean(np.isclose(y_pred, y_true_s1))
        print(f"  Característica '{agg:6s}' | shape scores={scores.shape} | Acerto S1 = {acc:5.2f}%")

print("\\n=== Adicionando Harmônicos (1f + 2f + 3f) no Periodograma FFT (S1, 9 canais) ===")
for agg in ["max", "mean", "median"]:
    y_pred_h, _ = classify_by_psd_voting(
        epochs_s1, target_freqs=BETA_SORTED_FREQS, threshold=0.08,
        agg_mode=agg, method="periodogram", n_harmonics=3
    )
    acc_h = 100.0 * np.mean(np.isclose(y_pred_h, y_true_s1))
    print(f"  Característica '{agg:6s}' (3 harmônicos) | Acerto S1 = {acc_h:5.2f}%")
"""),
        md_cell("""
### 2. Benchmark nos 10 Voluntários (`S1` a `S10`): Impacto dos Canais (`OZ` vs `3ch` vs `5ch` vs `9ch`) e Harmônicos
"""),
        code_cell("""
presets = ["oz", "3ch", "5ch", "9ch"]
voting_results = {p: {"1f": [], "3f": []} for p in presets}

for sub_id in range(1, 11):
    for p in presets:
        X_sub, y_f, _, _, meta = load_beta_subject(sub_id, channels=p, window="stim_latency")
        X_filt = butter_bandpass(butter_notch(X_sub, 50.0, fs=meta["sfreq"]), 8.0, 64.0, fs=meta["sfreq"])
        ep_sub = create_beta_epochs(X_filt, y_f, ch_names=meta["ch_names"], sfreq=meta["sfreq"])

        yp1, _ = classify_by_psd_voting(ep_sub, threshold=0.08, agg_mode="max", method="periodogram", n_harmonics=1)
        yp3, _ = classify_by_psd_voting(ep_sub, threshold=0.08, agg_mode="max", method="periodogram", n_harmonics=3)

        voting_results[p]["1f"].append(100.0 * np.mean(np.isclose(yp1, y_f)))
        voting_results[p]["3f"].append(100.0 * np.mean(np.isclose(yp3, y_f)))

print("Resumo de Acurácia Média ± Desvio Padrão (S1..S10 — 40 classes, chance = 2.50%):")
for p in presets:
    m1, s1 = np.mean(voting_results[p]["1f"]), np.std(voting_results[p]["1f"])
    m3, s3 = np.mean(voting_results[p]["3f"]), np.std(voting_results[p]["3f"])
    print(f"  Canais {p:>3s} | Fundamental (1f): {m1:5.2f}% ± {s1:5.2f}% | Com Harmônicos (1f+2f+3f): {m3:5.2f}% ± {s3:5.2f}%")

fig, ax = plt.subplots(figsize=(9, 4.8))
x_pos = np.arange(len(presets))
width = 0.35
means_1f = [np.mean(voting_results[p]["1f"]) for p in presets]
means_3f = [np.mean(voting_results[p]["3f"]) for p in presets]

b1 = ax.bar(x_pos - width/2, means_1f, width, label="Apenas Fundamental (1f)", color="#63b3ed")
b2 = ax.bar(x_pos + width/2, means_3f, width, label="Soma Harmônica (1f + 2f + 3f)", color="#2b6cb0")
ax.axhline(2.5, color="crimson", linestyle="--", linewidth=1, label="Chance Aleatória (2.5%)")
ax.set_xticks(x_pos)
ax.set_xticklabels(["1 Canal (OZ)", "3 Canais (O1,OZ,O2)", "5 Canais", "9 Canais Padrão"])
ax.set_ylabel("Acurácia Média S1..S10 (%)")
ax.set_title("Classificador por Isolamento de Frequência (Voto PSD) no Dataset BETA")
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.show()
"""),
    ]


def build_nb04() -> List[Dict[str, Any]]:
    return [
        md_cell("""
# 04 — Extração de Características Reais no BETA (`Narrow SNR`, `Wide SNR`, Espectrais e Temporais)

> **Adaptação de:** `SSVEP/beta_ml.ipynb`, `Focus/src/snr_feature.ipynb` e `SomeFeatures/04-features.ipynb` ([bneurd/tutoriais](https://github.com/bneurd/tutoriais)).

## O que foi aprimorado em relação ao rascunho original `beta_ml.ipynb`?
No notebook original do repositório `tutoriais`, os cálculos de `narrow-band SNR` e `wide-band SNR` haviam sido demonstrados sobre **dados aleatórios fictícios** (`np.random.normal(size=(160, 64, 750))`), forçando `estimated_background_noise = 1.0` para evitar logaritmo de números negativos (`NaN`).

Aqui aplicamos a formulação física exata sobre os **dados reais do Dataset BETA**:
1. **Ruído de Fundo Real (`psd_noise`):** Medido diretamente nos trechos sem estímulo de cada tentativa (`0.0 – 0.5 s` e `2.5 – 3.0 s`);
2. **Sinal Estimulado (`psd_stim`):** Medido na janela SSVEP estacionária (`0.63 – 2.5 s`, após os `130 ms` de latência visual) com `n_fft = 1250` (resolução exata de `0.2 Hz` cravada nos 40 alvos);
3. **4 Matrizes Espectrais `(160, C, 40)` $\\rightarrow$ `(160, C * 160)`:**
   * `narrow_snr` (dB): potência no alvo vs. frequências vizinhas (Eq. 3 do artigo BETA);
   * `wide_snr` (dB): potência harmônica vs. potência total 8–64 Hz (Eq. 4 do artigo BETA);
   * `basal_snr` (dB): potência estimulada vs. ruído basal medido antes/depois do estímulo;
   * `target_amp`: amplitude harmônica combinada;
4. **Características Clássicas (`SomeFeatures`):** `VAR`, `RMS`, `FMD` e `MMDF` por canal.
"""),
        code_cell(COMMON_HEADER_CODE + """
from src.bneurd_beta import (
    PROCESSED_DATA_DIR,
    butter_bandpass,
    butter_notch,
    extract_beta_ml_features,
    extract_classic_channel_features,
    load_beta_subject,
)

# 1. Carrega todos os 64 canais de S1 para comprovar a dimensionalidade (160, 10240) de tunning.ipynb
X_s1_64, y_freqs_s1, y_cls_s1, blocks_s1, meta_64 = load_beta_subject(1, channels="all", window="full")
X_s1_64_filt = butter_bandpass(butter_notch(X_s1_64, 50.0, fs=250), 8.0, 64.0, fs=250)

feats_64 = extract_beta_ml_features(X_s1_64_filt)
classic_64 = extract_classic_channel_features(X_s1_64_filt[:, :, 157:625])

print("Verificação das dimensões em 64 canais (conforme previsto em tunning.ipynb):")
print(" - narrow_snr    :", feats_64["narrow_snr"].shape, "| NaNs:", np.isnan(feats_64["narrow_snr"]).sum())
print(" - wide_snr      :", feats_64["wide_snr"].shape,   "| NaNs:", np.isnan(feats_64["wide_snr"]).sum())
print(" - basal_snr     :", feats_64["basal_snr"].shape,  "| NaNs:", np.isnan(feats_64["basal_snr"]).sum())
print(" - target_amp    :", feats_64["target_amp"].shape, "| NaNs:", np.isnan(feats_64["target_amp"]).sum())
print(" - feature_matrix:", feats_64["feature_matrix"].shape, "-> (160 amostras, 64 canais * 40 alvos * 4 = 10240)")
print(" - classic (VAR, RMS, FMD, MMDF):", classic_64.shape, "-> (160 amostras, 64 * 4 = 256)")
"""),
        md_cell("""
### 2. Extração nos 9 Canais Parieto-Occipitais e Visualização da Diagonal de Ressonância SSVEP
Ao observar a média de `narrow_snr` e `basal_snr` nos 9 canais visuais para cada classe real (0 a 39) vs. cada frequência candidata (8.0 a 15.8 Hz), vemos uma **diagonal nítida**, provando que cada estímulo eleva seletivamente a SNR na sua exata frequência!
"""),
        code_cell("""
X_s1_9, _, _, _, meta_9 = load_beta_subject(1, channels="9ch", window="full")
X_s1_9_filt = butter_bandpass(butter_notch(X_s1_9, 50.0, fs=250), 8.0, 64.0, fs=250)
feats_9 = extract_beta_ml_features(X_s1_9_filt)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
np.save(PROCESSED_DATA_DIR / "beta_s1_features_64ch.npy", feats_64["feature_matrix"])
np.save(PROCESSED_DATA_DIR / "beta_s1_features_9ch.npy", feats_9["feature_matrix"])
np.save(PROCESSED_DATA_DIR / "beta_s1_y_class.npy", y_cls_s1)
print("Matrizes de características salvas:", feats_9["feature_matrix"].shape)

# Calcula matriz média (40 classes reais x 40 frequências candidatas) sobre os 4 blocos e 9 canais
diag_narrow = np.zeros((40, 40))
diag_wide = np.zeros((40, 40))
for cls_idx in range(40):
    trial_mask = (y_cls_s1 == cls_idx)
    diag_narrow[cls_idx] = feats_9["narrow_snr"][trial_mask].mean(axis=(0, 1))
    diag_wide[cls_idx] = feats_9["wide_snr"][trial_mask].mean(axis=(0, 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

im0 = axes[0].imshow(diag_narrow, aspect="auto", origin="lower", cmap="magma",
                     extent=[8.0, 15.8, 8.0, 15.8])
axes[0].set_title("Narrow-band SNR Média (dB) — S1 (9 Canais Occipitais)")
axes[0].set_xlabel("Frequência Candidata Analisada (Hz)")
axes[0].set_ylabel("Frequência Real Estimulada (Hz)")
plt.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(diag_wide, aspect="auto", origin="lower", cmap="viridis",
                     extent=[8.0, 15.8, 8.0, 15.8])
axes[1].set_title("Wide-band SNR Média (dB, Harmônicos 1f-3f) — S1")
axes[1].set_xlabel("Frequência Candidata Analisada (Hz)")
axes[1].set_ylabel("Frequência Real Estimulada (Hz)")
plt.colorbar(im1, ax=axes[1], label="dB")

plt.tight_layout()
plt.show()
"""),
    ]


def build_nb05() -> List[Dict[str, Any]]:
    return [
        md_cell("""
# 05 — Visualização Grand-Average, Seleção de Características e Tuning de SVM no BETA

> **Adaptação de:** `SSVEP/tunning.ipynb` e `SomeFeatures/05-viz_selection.ipynb` ([bneurd/tutoriais](https://github.com/bneurd/tutoriais)).

Este notebook fecha o ciclo completo de preparação pedido pelo professor antes de iniciarmos os modelos profundos do projeto:
1. **Visualização Grand-Average (`S1` a `S10`):** Reproduz as células finais de `tunning.ipynb` tirando a média dos 10 participantes, dos 4 blocos e dos 9 eletrodos occipitais (`PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2`), removendo o trecho basal e plotando os espectros FFT com linha tracejada vermelha nos alvos;
2. **Projeção 2D com PCA (`05-viz_selection.ipynb`):** Visualiza a separabilidade espacial das classes reduzindo a dimensionalidade via PCA;
3. **Seleção de Características (`VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `RFE`) + `SVM`:** Compara todos os seletores sugeridos em `tunning.ipynb` usando validação cruzada por blocos (*Leave-One-Block-Out*, 4 folds);
4. **Tuning de Hiperparâmetros (`GridSearchCV`) e Comparativo de Canais (`64ch vs 9ch vs 5ch vs 3ch`) nos 10 Voluntários (`S1..S10`)!**
"""),
        code_cell(COMMON_HEADER_CODE + """
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.bneurd_beta import (
    BETA_SORTED_FREQS,
    CHANNEL_PRESETS,
    butter_bandpass,
    butter_notch,
    evaluate_selectors_and_svm,
    extract_beta_ml_features,
    load_beta_subject,
    tune_svm_hyperparameters,
)
"""),
        md_cell("""
### 1. Visualização Grand-Average SSVEP nos 10 Voluntários (`S1` a `S10`)
Adaptando fielmente `tunning.ipynb` (células 6 a 9) para os nossos arquivos locais `data/raw/S1.mat..S10.mat`:
"""),
        code_cell("""
# Empilha os 10 participantes já nos 9 eletrodos occipitais e sem a parte basal (0.63s a 2.5s)
group_trials = []
for sub_id in range(1, 11):
    X_sub, y_f, y_c, blk, meta = load_beta_subject(sub_id, channels="9ch", window="stim_latency")
    X_sub_filt = butter_bandpass(butter_notch(X_sub, 50.0, fs=250), 8.0, 64.0, fs=250)
    # Reorganiza (160, 9, T) -> (4 blocos, 40 alvos, 9 canais, T)
    group_trials.append(X_sub_filt.reshape(4, 40, 9, -1))

eeg_group = np.array(group_trials)  # (10 sujeitos, 4 blocos, 40 alvos, 9 canais, T)
eeg_avg = eeg_group.mean(axis=(0, 1, 3))  # Média dos sujeitos, blocos e 9 canais -> (40, T)
print("Shape empilhado (suj, blocos, alvos, canais, tempo):", eeg_group.shape)
print("Shape médio final (40 alvos, tempo):", eeg_avg.shape)

# Plota grade 3x3 com 9 frequências representativas distribuídas entre 8.0 e 15.8 Hz
sample_indices = [0, 5, 10, 15, 20, 25, 30, 35, 39]
freqs_fft = np.fft.rfftfreq(1250, d=1.0 / 250.0)
w_fft = np.abs(np.fft.rfft(eeg_avg, n=1250, axis=-1)) / eeg_avg.shape[-1]

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
for ax, idx_t in zip(axes.flatten(), sample_indices):
    target_hz = BETA_SORTED_FREQS[idx_t]
    ax.plot(freqs_fft, w_fft[idx_t], color="#1a365d", linewidth=1.2)
    ax.axvline(target_hz, linestyle="--", linewidth=1.1, color="crimson", label=f"Alvo {target_hz:.1f} Hz")
    ax.axvline(2 * target_hz, linestyle=":", linewidth=1.0, color="darkorange", label=f"2f {2*target_hz:.1f} Hz")
    ax.set_xlim(5, 45)
    ax.set_title(f"Alvo Estimulado: {target_hz:.1f} Hz", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

for ax in axes[-1, :]:
    ax.set_xlabel("Frequência (Hz)")
for ax in axes[:, 0]:
    ax.set_ylabel("Amplitude Média (uV)")

plt.suptitle("Espectro FFT Grand-Average (S1..S10, 4 Blocos, 9 Canais Occipitais)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""),
        md_cell("""
### 2. Visualização Bidimensional com PCA (`SomeFeatures/05-viz_selection.ipynb`)
Conforme sugerido no tutorial `05-viz_selection.ipynb`, reduzimos os múltiplos canais/características a 2 componentes principais via `PCA` para visualizar o agrupamento das classes (exibindo 6 frequências bem espaçadas do participante `S1` para clareza visual):
"""),
        code_cell("""
X_s1_9, y_f_s1, y_cls_s1, blocks_s1, _ = load_beta_subject(1, channels="9ch", window="full")
X_s1_9_filt = butter_bandpass(butter_notch(X_s1_9, 50.0, fs=250), 8.0, 64.0, fs=250)
feats_s1_9 = extract_beta_ml_features(X_s1_9_filt)["feature_matrix"]

subset_freqs = [8.0, 9.6, 11.2, 12.8, 14.4, 15.8]
mask_sub = np.isin(np.round(y_f_s1, 1), subset_freqs)
X_pca_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(feats_s1_9[mask_sub]))

fig, ax = plt.subplots(figsize=(8, 5.5))
for f_val in subset_freqs:
    pts = X_pca_2d[np.isclose(y_f_s1[mask_sub], f_val)]
    ax.scatter(pts[:, 0], pts[:, 1], s=70, alpha=0.85, label=f"{f_val:.1f} Hz")

ax.set_title("Dispersão PCA 2D das Características SNR/FFT — S1 (6 Alvos Exemplares)")
ax.set_xlabel("Componente Principal 1 (PC1)")
ax.set_ylabel("Componente Principal 2 (PC2)")
ax.legend(title="Estímulo")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
        md_cell("""
### 3. Seleção de Características (`VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `RFE`) + Classificador `SVM`
Comparamos em `S1` a matriz completa de **64 canais** (`10.240` características, onde muitos canais frontais/temporais contêm apenas ruído!) contra os seletores de características e a seleção espacial nos **9 canais occipitais** (`1.440` características):
"""),
        code_cell("""
X_s1_64, _, _, _, _ = load_beta_subject(1, channels="all", window="full")
X_s1_64_filt = butter_bandpass(butter_notch(X_s1_64, 50.0, fs=250), 8.0, 64.0, fs=250)
feats_s1_64 = extract_beta_ml_features(X_s1_64_filt)["feature_matrix"]

res_64 = evaluate_selectors_and_svm(feats_s1_64, y_cls_s1, blocks=blocks_s1, k_best=240, percentile=10, rfe_features=200)
res_9 = evaluate_selectors_and_svm(feats_s1_9, y_cls_s1, blocks=blocks_s1, k_best=240, percentile=25, rfe_features=200)

print("=== Resultados S1 — 64 Canais Completos (10.240 features) ===")
for k, v in res_64.items():
    print(f"  {k:32s}: {v:6.2f}%")

print("\\n=== Resultados S1 — 9 Canais Parieto-Occipitais (1.440 features) ===")
for k, v in res_9.items():
    print(f"  {k:32s}: {v:6.2f}%")
"""),
        md_cell("""
### 4. Tuning do `SVM` (`GridSearchCV`) e Avaliação nos 10 Voluntários (`S1..S10`) Variando Canais (`64 vs 9 vs 5 vs 3`)
Realizamos o ajuste fino de hiperparâmetros (`C`, `kernel`, `k`) e avaliamos todos os 10 voluntários baixados, conectando diretamente o tutorial preliminar do BNEURD à questão central da nossa pesquisa em OPT098!
"""),
        code_cell("""
best_params, best_acc_s1, _ = tune_svm_hyperparameters(feats_s1_9, y_cls_s1, blocks_s1, k_features=240)
print(f"Melhor configuração SVM após Tuning em S1 (9ch): {best_acc_s1:.2f}% | params={best_params}")

channel_configs = ["all", "9ch", "5ch", "3ch"]
svm_bench = {cfg: [] for cfg in channel_configs}

for sub_id in range(1, 11):
    X_all, _, y_c, blk, _ = load_beta_subject(sub_id, channels="all", window="full")
    X_all_filt = butter_bandpass(butter_notch(X_all, 50.0, fs=250), 8.0, 64.0, fs=250)

    for cfg in channel_configs:
        ch_idx = list(range(64)) if cfg == "all" else CHANNEL_PRESETS[cfg]
        f_mat = extract_beta_ml_features(X_all_filt[:, ch_idx, :])["feature_matrix"]
        _, best_sub_acc, _ = tune_svm_hyperparameters(f_mat, y_c, blk, k_features=min(240, f_mat.shape[1]))
        svm_bench[cfg].append(best_sub_acc)

print("\\n=== Benchmark Final SVM + SelectKBest (Tuned) nos 10 Sujeitos (S1..S10) ===")
for cfg in channel_configs:
    arr = np.array(svm_bench[cfg])
    print(f"  Configuração {cfg:>4s} | Média: {arr.mean():6.2f}% ± {arr.std():5.2f}% | Min: {arr.min():5.2f}% | Max: {arr.max():5.2f}%")

fig, ax = plt.subplots(figsize=(9, 5))
labels_cfg = ["64 Canais (Todos)", "9 Canais (Padrão)", "5 Canais (Reduzido)", "3 Canais (Mínimo)"]
means_cfg = [np.mean(svm_bench[c]) for c in channel_configs]
stds_cfg = [np.std(svm_bench[c]) for c in channel_configs]

bars = ax.bar(labels_cfg, means_cfg, yerr=stds_cfg, capsize=5, color=["#718096", "#2b6cb0", "#319795", "#dd6b20"], alpha=0.9)
ax.axhline(2.5, color="crimson", linestyle="--", linewidth=1, label="Chance Aleatória (2.5% — 40 classes)")
ax.set_ylabel("Acurácia Leave-One-Block-Out (%)")
ax.set_ylim(0, 100)
ax.set_title("Desempenho SVM + Seleção de Características no Dataset BETA (S1..S10)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bar, m in zip(bars, means_cfg):
    ax.text(bar.get_x() + bar.get_width()/2, m + 3.0, f"{m:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()
"""),
    ]


def main() -> None:
    notebooks_spec: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("01_intro_e_carregamento_beta.ipynb", build_nb01()),
        ("02_preprocessamento_e_mne_beta.ipynb", build_nb02()),
        ("03_classificador_por_frequencia_beta.ipynb", build_nb03()),
        ("04_extracao_caracteristicas_snr_beta.ipynb", build_nb04()),
        ("05_selecao_tunning_svm_beta.ipynb", build_nb05()),
    ]

    summary_data: Dict[str, Any] = {}

    for filename, cells in notebooks_spec:
        print(f"[EXECUTANDO] {filename} ...")
        ns = execute_notebook_cells(cells, filename)
        out_file = NOTEBOOKS_OUT_DIR / filename
        save_notebook(cells, out_file)
        print(f"[OK] {filename} salvo ({out_file.stat().st_size / 1024:.1f} KB)")

        if "voting_results" in ns:
            summary_data["voting_results"] = {
                k: {
                    m: {
                        "mean": round(float(sum(vals) / len(vals)), 2),
                        "std": round(float( (sum((x - sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5 ), 2),
                    }
                    for m, vals in v.items()
                }
                for k, v in ns["voting_results"].items()
            }
        if "res_64" in ns:
            summary_data["s1_selectors_64ch"] = {k: round(float(v), 2) for k, v in ns["res_64"].items()}
        if "res_9" in ns:
            summary_data["s1_selectors_9ch"] = {k: round(float(v), 2) for k, v in ns["res_9"].items()}
        if "svm_bench" in ns:
            import numpy as _np
            summary_data["svm_bench_s1_s10"] = {
                k: {
                    "mean": round(float(_np.mean(v)), 2),
                    "std": round(float(_np.std(v)), 2),
                    "per_subject": [round(float(x), 2) for x in v],
                }
                for k, v in ns["svm_bench"].items()
            }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumo empírico salvo em: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
