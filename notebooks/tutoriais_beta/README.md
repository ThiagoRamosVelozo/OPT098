# Adaptação dos Tutoriais BNEURD (`bneurd/tutoriais`) para o Dataset BETA

Este diretório contém a adaptação completa e executada do repositório didático **[bneurd/tutoriais](https://github.com/bneurd/tutoriais)** (especialmente as trilhas `SSVEP/`, `SomeFeatures/` e `Focus/`) para a base de dados oficial do nosso projeto: o **Dataset BETA** (*Liu et al., 2020* — 64 canais, 250 Hz, 4 blocos, 40 alvos de `8.0` a `15.8 Hz`).

Todas as rotinas reutilizáveis estão centralizadas em [`src/bneurd_beta.py`](../../src/bneurd_beta.py), e os 5 notebooks podem ser regenerados e reexecutados do zero a qualquer momento com:

```powershell
.\.venv\Scripts\python.exe scripts\build_and_run_tutoriais_beta.py
```

---

## 1. Mapeamento Direto: `bneurd/tutoriais` $\rightarrow$ `notebooks/tutoriais_beta/`

| Notebook Adaptado (Nossa Base BETA) | Origem em `bneurd/tutoriais` | Principais Adaptações Realizadas para o BETA |
| :--- | :--- | :--- |
| **[`01_intro_e_carregamento_beta.ipynb`](./01_intro_e_carregamento_beta.ipynb)** | `SSVEP/intro.ipynb`<br>`SSVEP/load_single_avi.ipynb`<br>`SSVEP/load_multi_avi_exemplo.ipynb` | Transição do dataset AVI (1 canal `Oz`, 512 Hz, 7 alvos) para o **BETA** (`S1.mat` a `S10.mat`, 64 canais 10-20, 250 Hz, 40 alvos espaçados em `0.2 Hz`), ordenação por frequência (`8.0–15.8 Hz`), padronização em `(160, 64, 750)` e exportação `.npy`. |
| **[`02_preprocessamento_e_mne_beta.ipynb`](./02_preprocessamento_e_mne_beta.ipynb)** | `SSVEP/preprocessing.ipynb`<br>`SSVEP/beta_ml.ipynb` *(seção MNE)* | Filtros Butterworth de fase zero (`bandpass`, `lowpass`, `highpass` e `notch` em **50 Hz**), criação do `mne.EpochsArray` com 40 eventos, comparação via `scipy.fft` entre janelas **sem estímulo** (`0.0–0.5s` e `2.5–3.0s`) vs. **com estímulo SSVEP** (`0.5–2.5s` / `0.63–2.5s` sem latência de `130 ms`) e salvamento em `.fif`. |
| **[`03_classificador_por_frequencia_beta.ipynb`](./03_classificador_por_frequencia_beta.ipynb)** | `SSVEP/classifier_per_freq.ipynb` | Ajuste do limiar espectral de `0.25 Hz` (AVI) para **`0.08 Hz`** (compatível com o passo de `0.2 Hz` entre os 40 alvos do BETA), comparação entre `multitaper` e periodograma FFT com *zero-padding*, extratores `max`/`mean`/`median`, variação de canais (`OZ`, `3ch`, `5ch`, `9ch`) e soma harmônica ($1f + 2f + 3f$). |
| **[`04_extracao_caracteristicas_snr_beta.ipynb`](./04_extracao_caracteristicas_snr_beta.ipynb)** | `SSVEP/beta_ml.ipynb`<br>`Focus/src/snr_feature.ipynb`<br>`SomeFeatures/04-features.ipynb` | Substituição dos dados aleatórios fictícios (`np.random.normal`) pelos sinais reais do BETA, estimativa real do ruído de fundo nos trechos `[0.0–0.5s]` e `[2.5–3.0s]`, cálculo sem `NaN` de `narrow-band SNR`, `wide-band SNR`, `basal SNR` e amplitude harmônica gerando a matriz `(160, 10240)` em 64 canais e `(160, 1440)` em 9 canais, além de `VAR, RMS, FMD, MMDF`. |
| **[`05_selecao_tunning_svm_beta.ipynb`](./05_selecao_tunning_svm_beta.ipynb)** | `SSVEP/tunning.ipynb`<br>`SomeFeatures/05-viz_selection.ipynb` | Espectro FFT *Grand-Average* sobre `S1..S10` nos 9 eletrodos occipitais (`PZ, PO5, PO3, POZ, PO4, PO6, O1, OZ, O2`), projeção 2D via `PCA`, avaliação *Leave-One-Block-Out* (4 folds) de `VarianceThreshold`, `SelectKBest`, `SelectPercentile` e `RFE` com `SVM`, *tuning* com `GridSearchCV` e comparativo `64ch vs 9ch vs 5ch vs 3ch`. |

---

## 2. Principais Resultados Empíricos Obtidos (40 Classes — Chance Aleatória = 2,50%)

### 2.1. Classificador por Isolamento de Frequência e Voto (`Notebook 03`)
* Em janelas curtas (`1,87 s`), o `multitaper` padrão do MNE possui largura de banda $\approx 2,1\text{ Hz}$, misturando alvos vizinhos separados por `0,2 Hz` (`3,75%` em `S1`).
* Com **Periodograma FFT com Zero-Padding** (`threshold = 0,08 Hz`), o acerto sem treinamento em `S1` salta para **53,12%** apenas na fundamental ($1f$) e **82,50%** ao somar os três primeiros harmônicos ($1f + 2f + 3f$)!
* Na média dos 10 sujeitos (`S1..S10`), somar harmônicos eleva o acerto em `OZ` de **38,56% $\rightarrow$ 51,69%**. Além disso, tirar a média simples não ponderada de 9 canais cai levemente para **45,25%**, evidenciando na prática por que algoritmos com **filtragem espacial** (FBCCA) e **atenção por canal** (DB-AFDNet) são indispensáveis.

### 2.2. Impacto da Seleção de Características e Seleção Espacial no `SVM` (`Notebook 05` — Participante `S1`)

| Método de Seleção + Classificador SVM (4-Fold por Bloco) | 64 Canais (`10.240` features) | 9 Canais Occipitais (`1.440` features) |
| :--- | :---: | :---: |
| **SVM Puro (Sem Seleção)** | 28,75% | 63,13% |
| **VarianceThreshold + SVM** | 28,75% | 63,13% |
| **RFE (`n=200`) + SVM** | 70,00% | 73,75% |
| **SelectPercentile + SVM** | 75,62% *(10%)* | **81,88%** *(25%)* |
| **SelectKBest (`k=240`) + SVM** | **79,38%** | 80,62% |

> **Conclusão Prática:** Nos 64 canais brutos, eletrodos frontais e temporais introduzem forte ruído (maldição da dimensionalidade: `28,75%`). A seleção de características (`SelectKBest` / `SelectPercentile` / `RFE`) recupera mais de **+50 pontos percentuais** de acurácia (`79,38%`), e restringir previamente aos **9 eletrodos parieto-occipitais** atinge **81,88%** com 7× menos variáveis!

### 2.3. Prévia da Redução de Canais nos 10 Voluntários (`S1..S10` — `SVM Tuned`)
* **64 Canais (`all`):** `45,94% ± 22,90%` (máx. `79,38%`)
* **9 Canais (`9ch`):** `42,12% ± 25,89%` (máx. `80,62%`)
* **5 Canais (`5ch`):** `42,06% ± 24,98%` (máx. `80,62%` — praticamente sem perda frente a 9 canais!)
* **3 Canais (`3ch`):** `37,94% ± 27,15%` (máx. `80,00%`)
