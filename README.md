# OPT098 - SSVEP-BCI Decodificação Inter-Sujeito

Este repositório contém o código, experimentos e pipelines do projeto para a disciplina **OPT098 - Tópicos Especiais em Sistemas de Computação**.

**Título do Trabalho:** *Estudo Comparativo de Metodologias e Recursos em SSVEP-BCI: Da Modelagem Espectral à Aprendizagem em Variedades*  
**Autores:**
- Thiago Ramos Velozo (RA: 2135760 | GitHub: [@ThiagoRamosVelozo](https://github.com/ThiagoRamosVelozo))
- Victor Angelo Souza Santos (RA: 2021064 | GitHub: [@Victor476](https://github.com/Victor476))

---

## 1. Estrutura do Repositório

```text
OPT098/
├── data/
│   ├── raw/                 # Dados brutos do dataset BETA (S*.mat, Description.pdf, etc.)
│   └── processed/           # Tensores pré-processados e filtrados
├── src/
│   ├── data/                # Ingestão, fatiamento, filtros e seleção espacial
│   ├── models/              # Arquiteturas (FBCCA baseline, DB-AFDNet, etc.)
│   └── utils/               # Métricas (ITR, acurácia), gráficos e configurações
├── models/                  # Checkpoints e pesos treinados (.pt, .pth)
├── experiments/
│   ├── logs/                # Logs de treinamento e validação LOSO
│   └── results/             # Matrizes de confusão, CSVs e relatórios
├── notebooks/               # Análise exploratória e visualizações
├── scripts/
│   ├── verify_env.py        # Validação do ambiente Python e dependências
│   ├── download_beta.py     # Script de download automatizado do dataset BETA
│   └── verify_beta_data.py  # Script de validação e integridade dos arquivos .mat
├── requirements.txt         # Dependências do projeto (PyTorch, MNE, SciPy, etc.)
└── .gitignore               # Arquivos ignorados (dados pesados, .venv, modelos)
```

---

## 2. Configuração do Ambiente Virtual (Issue 01)

### Criar e Ativar o Ambiente Virtual
No Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalar Dependências
```powershell
pip install -r requirements.txt
```

### Testar a Instalação (Critério de Aceitação)
```powershell
python -c "import torch, scipy, mne; print('OK')"
```
Ou execute o script detalhado de verificação:
```powershell
python scripts/verify_env.py
```

---

## 3. Download e Validação do Dataset BETA (Issue 02)

O dataset **BETA** (*Benchmark database toward SSVEP-BCI Application*) é composto por gravações de EEG de 64 canais, 40 alvos e 4 blocos amostrados a 250 Hz (Liu et al., 2020).

### Baixar Amostras (Sujeitos S1 a S10)
```powershell
python scripts/download_beta.py --num-subjects 10
```

### Validar Integridade e Estrutura dos Tensores
```powershell
python scripts/verify_beta_data.py
```
O script checa:
- Presença e carregamento dos arquivos `.mat` via `scipy.io`.
- Presença da variável `data` e dimensão do tensor de EEG `(64 canais, 750 amostras, 4 blocos, 40 alvos)`.
- Ausência de valores corrompidos (`NaN` ou `Inf`).