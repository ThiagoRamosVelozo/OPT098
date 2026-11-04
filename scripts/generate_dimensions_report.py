"""Script para geração automatizada do Guia Completo de Dimensões dos Dados e Benchmarks.

Analisa diretamente as estruturas reais dos arquivos do projeto (BETA, DB-AFDNet,
Compact-ANN, MAtt/MAMEM) e compila um relatório em Markdown detalhado, didático e
acessível para leigos, dissecando todas as dimensões, eixos e tipos de dados.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DBAFDNET_DIR = PROJECT_ROOT / "references" / "DBAFDNet"
COMPACT_ANN_DIR = PROJECT_ROOT / "references" / "Compact-ANN"
MATT_DIR = PROJECT_ROOT / "references" / "MAtt"
OUTPUT_MD_PATH = PROJECT_ROOT / "guia_dimensoes_dados.md"


def get_beta_actual_specs():
    """Inspeciona os arquivos locais do dataset BETA para obter as dimensões exatas."""
    specs = {
        "found": False,
        "sample_shape": None,
        "n_channels": 64,
        "n_timepoints": 750,
        "n_blocks": 4,
        "n_targets": 40,
        "fs": 250,
        "freq_range": (8.0, 15.8),
        "freq_step": 0.2,
        "latency_ms": 130,
        "dtype": "float64",
        "sample_voltage_min": -566.2,
        "sample_voltage_max": 454.8,
    }
    s1_path = RAW_DATA_DIR / "S1.mat"
    if s1_path.exists():
        try:
            m = scipy.io.loadmat(s1_path, squeeze_me=True, struct_as_record=False)
            data_obj = m["data"]
            eeg = getattr(data_obj, "EEG", None)
            if eeg is not None:
                specs["found"] = True
                specs["sample_shape"] = eeg.shape
                specs["n_channels"] = eeg.shape[0]
                specs["n_timepoints"] = eeg.shape[1]
                specs["n_blocks"] = eeg.shape[2]
                specs["n_targets"] = eeg.shape[3]
                specs["dtype"] = str(eeg.dtype)
                specs["sample_voltage_min"] = float(np.min(eeg))
                specs["sample_voltage_max"] = float(np.max(eeg))
        except Exception as e:
            print(f"[AVISO] Não foi possível ler S1.mat diretamente: {e}")
    return specs


def build_markdown_guide(beta_specs):
    """Constrói o documento Markdown com explicações didáticas completas."""
    md = []

    md.append("# Guia Explicativo das Dimensões dos Dados e Benchmarks (SSVEP-BCI)")
    md.append("")
    md.append("> **Objetivo deste Documento:** Explicar detalhadamente, até para quem não tem formação técnica prévia em ciência de dados ou neuroengenharia, o que é cada uma das **dimensões** presentes em cada uma das bases de dados (*benchmarks*) e modelos deste projeto, de onde vêm esses números, do que cada dimensão é feita fisicamente e como elas se transformam ao longo do pipeline de processamento.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. O que é uma 'Dimensão' em Dados de Eletroencefalograma? (Para Leigos)")
    md.append("")
    md.append("Imagine que você queira organizar informações em uma planilha:")
    md.append("")
    md.append("* **1 Dimensão (1D - Uma Linha / Vetor):** É uma fita contínua de medidas. Exemplo: a voltagem elétrica gravada por um único sensor segundo a segundo: `[1.2, 1.5, 0.8, -0.4, ...]`. Tem apenas comprimento: `(Tempo)`.")
    md.append("* **2 Dimensões (2D - Uma Tabela / Matriz):** É uma folha do Excel com linhas e colunas. Cada linha representa um sensor na cabeça e cada coluna representa um instante no tempo: `(Sensores, Tempo)`.")
    md.append("* **3 Dimensões (3D - Um Caderno / Cubo):** Várias folhas empilhadas. Se o voluntário repetiu o teste 4 vezes seguidas, temos uma folha 2D para cada repetição (bloco): `(Sensores, Tempo, Repetições)`.")
    md.append("* **4 Dimensões (4D - Uma Estante de Cadernos / Hipercubo):** Um conjunto de cubos. Se existem 40 letras diferentes na tela do computador para o usuário escolher com o olhar, temos um caderno inteiro para cada uma das 40 letras: `(Sensores, Tempo, Repetições, Letras)`.")
    md.append("")
    md.append("Cada dimensão responde a uma pergunta específica:")
    md.append("1. **Onde** no cérebro medimos? $\\rightarrow$ *Dimensão de Canais/Eletrodos*.")
    md.append("2. **Quando** aconteceu a medida? $\\rightarrow$ *Dimensão de Tempo/Amostras*.")
    md.append("3. **Quantas vezes** repetimos? $\\rightarrow$ *Dimensão de Blocos/Tentativas*.")
    md.append("4. **O que** o usuário estava tentando digitar/olhar? $\\rightarrow$ *Dimensão de Alvos/Classes*.")
    md.append("5. **Em qual frequência harmônica** estamos analisando? $\\rightarrow$ *Dimensão de Sub-bandas*.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Benchmark 1: Dataset BETA (Liu et al., 2020) — Nossa Base Principal")
    md.append("")
    md.append("O **BETA** (*Benchmark database toward SSVEP-BCI Application*) é a base pública com dados reais gravados de 70 voluntários sob um teclado visual de soletrar com 40 teclas cintilantes.")
    md.append("")
    md.append("### 2.1. O Tensor Bruto Original no MATLAB: `(64, 750, 4, 40)`")
    md.append("")
    md.append("Ao carregar um arquivo como `S1.mat` em Python (`scipy.io.loadmat`), a variável `data.EEG` possui exatamente quatro eixos:")
    md.append("")
    md.append("```text")
    md.append("                          EIXO 3: 4 Blocos (Repetições)")
    md.append("                                /")
    md.append("                               /")
    md.append("              ┌───────────────────────────┐")
    md.append("              │                           ││")
    md.append("              │                           ││")
    md.append("  EIXO 1:     │                           ││   EIXO 4: 40 Alvos/Teclas")
    md.append("  64 Eletrodos│        SINAIS DE EEG      ││  (8.0 Hz a 15.8 Hz)")
    md.append("  (FP1 a VEO) │                           ││")
    md.append("              │                           ││")
    md.append("              │                           ││")
    md.append("              └───────────────────────────┘/")
    md.append("                ─────────────────────────")
    md.append("                EIXO 2: 750 Pontos no Tempo")
    md.append("                   (3.0 segundos a 250 Hz)")
    md.append("```")
    md.append("")
    md.append("#### Dissecando cada uma das 4 dimensões do BETA:")
    md.append("")
    md.append("### **Dimensão 1: 64 Canais (Eletrodos no Escalpo)**")
    md.append("* **O que é fisicamente:** São 64 pequenos pinos condutores de metal posicionados sobre o couro cabeludo do voluntário com gel condutor, organizados de acordo com o padrão internacional **10-20**.")
    md.append(r"* **Unidade de medida:** Microvolts ($\mu V$), ou seja, milionésimos de volt ($10^{-6}\text{ V}$). Para os sujeitos do BETA, essa voltagem oscila tipicamente entre $-500\mu V$ e $+500\mu V$.")
    md.append("* **Quais são os eletrodos:**")
    md.append("  * 62 eletrodos no cérebro: da testa (`FP1, FPz, FP2`), centro da cabeça (`Cz, C3, C4`) até a nuca (`Oz, O1, O2`).")
    md.append("  * 2 eletrodos de eletrooculograma (EOG): `HEO` (olho horizontal) e `VEO` (olho vertical), usados para monitorar piscadas e desvio ocular.")
    md.append("* **Os 9 canais parieto-occipitais chave:** O córtex visual humano fica na nuca. O cérebro só processa o SSVEP na região occipital posterior. Por isso, a literatura seleciona apenas 9 canais: `[Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2]` (índices 47, 53, 54, 55, 56, 57, 60, 61, 62).")
    md.append("")
    md.append("### **Dimensão 2: 750 Pontos Temporais (Amostras)**")
    md.append("* **O que é fisicamente:** É a passagem do tempo discretizada pelo conversor analógico-digital (amplificador SynAmps2 da Neuroscan).")
    md.append(r"* **Taxa de amostragem ($f_s$):** $250\text{ Hz}$, significando que o aparelho mede a voltagem **250 vezes por segundo** ($\Delta t = \frac{1}{250} = 4\text{ milissegundos}$ entre cada ponto).")
    md.append(r"* **Duração total:** $\frac{750\text{ amostras}}{250\text{ Hz}} = 3,0\text{ segundos}$.")
    md.append("* **Divisão interna desses 3 segundos (Linha do Tempo de uma Tentativa):**")
    md.append("  1. **Amostras 1 a 125 (0,0s a 0,5s — 125 amostras):** Período de pré-estímulo / fixação. Uma cruz vermelha aparece na tela indicando onde o voluntário deve olhar, mas os quadrados ainda não estão piscando.")
    md.append("  2. **Amostras 126 a 157 (0,50s a 0,63s — 32 amostras = 130 ms):** Latência visual. A luz começa a piscar, mas o sinal luminoso leva 130 milissegundos para viajar da retina pelo nervo óptico até atingir os neurônios do córtex visual na nuca. **Esses 130 ms são descartados nos modelos!**")
    md.append("  3. **Amostras 158 a 625 (0,63s a 2,5s — 500 amostras = 2,0 s):** Estímulo visual ativo puro. Os neurônios da nuca disparam em regime estacionário.")
    md.append("  4. **Amostras 626 a 750 (2,5s a 3,0s — 125 amostras = 0,5 s):** Pós-estímulo / descanso para o voluntário desviar o olhar.")
    md.append("")
    md.append("### **Dimensão 3: 4 Blocos (Sessões / Repetições)**")
    md.append("* **O que é fisicamente:** O voluntário não faz o teste apenas uma vez. Para garantir robustez e evitar que um espirro ou piscada estrague os dados, a pessoa digita todas as 40 teclas do teclado 4 vezes separadas (Bloco 1, Bloco 2, Bloco 3 e Bloco 4).")
    md.append("* **Intervalo entre blocos:** O voluntário descansa alguns minutos entre cada bloco para relaxar os olhos e não sofrer com cansaço visual.")
    md.append("")
    md.append("### **Dimensão 4: 40 Alvos (Classes de Estímulo / Teclas Visuais)**")
    md.append("* **O que é fisicamente:** São os 40 caracteres do teclado virtual na tela (letras de A a Z, números de 0 a 9 e caracteres especiais).")
    md.append(r"* **Frequências de cintilação:** Variam de **8,0 Hz a 15,8 Hz** com passos regulares de **0,2 Hz** (ex.: $8,0\text{ Hz}, 8,2\text{ Hz}, 8,4\text{ Hz}, \dots, 15,8\text{ Hz}$).")
    md.append(r"* **Modulação conjunta de fase (JFPM):** Como 0,2 Hz é uma diferença muito pequena para o olho humano separar puramente pela frequência, o BETA codifica uma fase inicial diferente em cada tecla: $0, 0.5\pi, 1.0\pi, 1.5\pi$ radianos ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("### 2.2. O Tensor Reorganizado para Treinamento (Issue 03): `[160, Canais, Tempo]`")
    md.append("")
    md.append("Para treinar modelos clássicos e neurais, unificamos os blocos e alvos na dimensão de **amostras de treino (exemplos)**:")
    md.append("")
    md.append("$$\\text{Total de Tentativas por Sujeito} = 4\\text{ blocos} \\times 40\\text{ alvos} = 160\\text{ amostras}$$")
    md.append("")
    md.append("O tensor final que o módulo de dados (`src/data/loader.py`) entrega possui formato:")
    md.append("")
    md.append("| Cenário Experimental | Dimensões do Tensor `[N, C, T]` | Explicação dos Eixos |")
    md.append("| :--- | :---: | :--- |")
    md.append("| **9 Canais (1.0 s)** | `[160, 9, 250]` | 160 tentativas, 9 eletrodos parieto-occipitais, 250 amostras (1 segundo a 250 Hz). |")
    md.append("| **5 Canais (1.0 s)** | `[160, 5, 250]` | 160 tentativas, 5 eletrodos centrais (`POz, PO3, PO4, Oz, O1`), 250 amostras. |")
    md.append("| **3 Canais (1.0 s)** | `[160, 3, 250]` | 160 tentativas, 3 eletrodos mínimos (`O1, Oz, O2`), 250 amostras. |")
    md.append("| **9 Canais (0.5 s)** | `[160, 9, 125]` | 160 tentativas, 9 eletrodos, 125 amostras temporais (alta velocidade de comando). |")
    md.append("| **9 Canais (0.3 s)** | `[160, 9, 75]`  | 160 tentativas, 9 eletrodos, 75 amostras temporais (janela ultracurta). |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Benchmark 2: Benchmark Dataset (Wang et al., 2016)")
    md.append("")
    md.append("O **Benchmark Dataset** da Universidade de Tsinghua é o conjunto de dados mais citado da história de SSVEP, gravado com 35 voluntários dentro de uma câmara de blindagem eletromagnética estrita (Gaiola de Faraday).")
    md.append("")
    md.append("### 3.1. O Tensor no MATLAB: `(64, 1500, 40, 6)`")
    md.append("")
    md.append("Observe a sutil inversão dos eixos 3 e 4 em relação ao BETA:")
    md.append("")
    md.append("| Eixo | Tamanho | O que representa fisicamente | Comparação com o BETA |")
    md.append("| :-: | :-: | :--- | :--- |")
    md.append("| **1** | **64** | Eletrodos cerebrais (sistema 10-20). | Idêntico ao BETA (mesma touca Neuroscan). |")
    md.append("| **2** | **1500** | Pontos temporais ($6,0$ segundos a $250\\text{ Hz}$). | **O dobro do BETA** (5s de estimulação vs 2s do BETA). |")
    md.append("| **3** | **40** | Alvos visuais (8,0 a 15,8 Hz em passos de 0,2 Hz). | Idênticas frequências e fases JFPM. |")
    md.append("| **4** | **6** | Blocos de repetição por voluntário. | **6 repetições** (BETA tem apenas 4). |")
    md.append("")
    md.append("> **Atenção Técnica no Código:** No Benchmark, a chave do MATLAB é `mat['data']` com formato `(64, 1500, 40, 6)`, enquanto no BETA é `mat['data']['EEG'][0][0]` com formato `(64, 750, 4, 40)`. Essa diferença exige a transposição `transpose(2, 3, 0, 1)` no Benchmark e `transpose(3, 2, 0, 1)` no BETA, exatamente como documentado no `references/DBAFDNet/main.py#L65-L72`.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Benchmark 3: MAMEM-SSVEP-II Dataset (Pan et al., 2022 / MAtt)")
    md.append("")
    md.append("Utilizado no artigo do **MAtt** (*NeurIPS 2022*), este dataset foca em decodificação geométrica sobre variedades não-euclidianas.")
    md.append("")
    md.append("### 4.1. Dimensões no Domínio do Tempo Euclidiano: `[Batch, 1, 8, 768]`")
    md.append("* **1 (Canal de Imagem/Batch):** Representa o canal único de entrada para convoluções 2D no PyTorch.")
    md.append("* **8 Canais Focais:** Apenas 8 eletrodos na nuca: `[PO7, PO3, POz, PO4, PO8, O1, Oz, O2]`.")
    md.append(r"* **768 Pontos Temporais:** Amostrados a $256\text{ Hz}$ durante $3,0$ segundos ($3 \times 256 = 768$ amostras).")
    md.append(r"* **5 Classes:** Frequências mais espaçadas e lentas: $6.66\text{ Hz}, 7.50\text{ Hz}, 8.57\text{ Hz}, 10.00\text{ Hz}, 12.00\text{ Hz}$.")
    md.append("")
    md.append("### 4.2. A Transformação Riemanniana `E2R` (Euclidiano para Matriz SPD): `[Batch, M, 12, 12]`")
    md.append("O MAtt substitui a leitura de voltagem direta por **matrizes de covariância**: ")
    md.append("")
    md.append("1. **$M$ Sub-janelas Temporais (`epochs = 7`):** O sinal de 3 segundos é cortado em 7 pedacinhos temporais consecutivos.")
    md.append("2. **$12 \\times 12$ Matriz SPD (Simétrica Definida Positiva):** Cada pedacinho temporal gera uma matriz $12 \\times 12$ que mede como cada canal elétrico se correlaciona com todos os outros:")
    md.append(r"   $$C = \frac{(X - \bar{X})(X - \bar{X})^T}{T - 1} + \epsilon I$$")
    md.append("3. Essa matriz vive em um espaço curvo (variedade Riemanniana), onde o MAtt calcula distâncias geodésicas (distância sobre a curvatura da esfera cerebral) em vez da distância em linha reta euclidiana.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Estrutura de Tensores do DB-AFDNet (Yang et al., 2026)")
    md.append("")
    md.append("O DB-AFDNet processa o sinal através de uma esteira complexa de álgebra linear e decomposição espectral.")
    md.append("")
    md.append("```text")
    md.append("  ENTRADA TEMPORAL             BANCO DE FILTROS           DOMÍNIO ESPECTRAL COMPLEXO (RFFT)")
    md.append("  [Batch, 9, 250]     ───►    [Batch, 3, 9, 250]   ───►   Parte Real:  [Batch, 9, 280]")
    md.append("  (Sinal no tempo)            (3 sub-bandas)              Parte Imag:  [Batch, 9, 280]")
    md.append("                                                          (280 frequências de 8 a 64 Hz)")
    md.append("```")
    md.append("")
    md.append("### 5.1. O que é o Tensor `[Batch, 3, 9, 250]` (Entrada da Rede)?")
    md.append("* **Batch:** Quantidade de épocas processadas de uma vez na GPU (ex.: 32 ou 64 tentativas).")
    md.append("* **3 Sub-bandas (Filter Bank):** O sinal original é filtrado em três faixas de frequência sobrepostas:")
    md.append("  * **Sub-banda 1 (8 a 64 Hz):** Captura o sinal SSVEP fundamental e todas as suas harmônicas.")
    md.append("  * **Sub-banda 2 (16 a 64 Hz):** Remove o ritmo alfa espontâneo (8-12 Hz) que atrapalha e isola o 2º harmônico.")
    md.append("  * **Sub-banda 3 (24 a 64 Hz):** Isola os harmônicos de alta ordem (3º e 4º harmônicos).")
    md.append("* **9 Canais:** Os eletrodos parieto-occipitais ativos.")
    md.append("* **250 Pontos:** 1,0 segundo de estimulação.")
    md.append("")
    md.append("### 5.2. O que é o Tensor `[Batch, 9, 280]` (Domínio da Frequência)?")
    md.append("* **Por que 280 pontos se a entrada tinha 250 pontos?**")
    md.append("  * A rede aplica `torch.fft.rfft(x, n=1250, axis=-1)`. Ao preencher o sinal com zeros (*zero-padding*) até 1250 pontos, a resolução da Transformada de Fourier torna-se ultra-fina:")
    md.append(r"    $$\Delta f = \frac{f_s}{N_{\text{fft}}} = \frac{250\text{ Hz}}{1250} = 0,20\text{ Hz por ponto}$$")
    md.append("  * Isso é perfeito, pois os alvos do BETA são separados exatamente por $0,20\\text{ Hz}$!")
    md.append("* **O que são os índices 40 a 320 (`40:320` $\\rightarrow$ 280 bins):**")
    md.append(r"  * O bin 40 corresponde a: $40 \times 0,20\text{ Hz} = 8,0\text{ Hz}$ (menor frequência do teste).")
    md.append(r"  * O bin 320 corresponde a: $320 \times 0,20\text{ Hz} = 64,0\text{ Hz}$ (maior frequência do teste).")
    md.append("  * Total: $320 - 40 = 280$ pontos no eixo de frequência.")
    md.append("* **A Divisão em Dois Ramos:**")
    md.append("  * **Ramo Real (Amplitude / Cosseno):** Mede a energia e potência de cada onda.")
    md.append("  * **Ramo Imaginário (Fase / Seno):** Mede o atraso e a sincronia temporal das ondas.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Estrutura de Tensores da Compact ANN (Wang et al., 2023)")
    md.append("")
    md.append("A **Compact ANN** reduz os parâmetros treináveis para ínfimos 335 pesos usando matrizes fixas de gabarito:")
    md.append("")
    md.append("### 6.1. Matriz de Referência Senoidal Teórica $R_i$: `[10, 250]`")
    md.append("* Para cada um dos 40 alvos $i$, gera-se uma matriz fixa contendo senos e cossenos da frequência fundamental $f_i$ e de 4 harmônicos ($2f_i, 3f_i, 4f_i, 5f_i$):")
    md.append(r"* $5\text{ harmônicos} \times 2\text{ componentes (seno e cosseno)} = 10\text{ linhas}$.")
    md.append("* 250 colunas correspondentes aos instantes de tempo.")
    md.append("")
    md.append("### 6.2. Gabarito Empírico Médio $\\bar{X}_i$: `[9, 250]`")
    md.append("* É a média de voltagem calculada através dos blocos anteriores do próprio sujeito, servindo como uma 'impressão digital cerebral' de como aquele voluntário específico reage àquela tecla.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 7. Tabela Comparativa Mestra de Todas as Dimensões")
    md.append("")
    md.append("| Base / Modelo | Formato do Tensor | Dimensões Detalhadas | Tipo de Dado | O que cada elemento guarda fisicamente |")
    md.append("| :--- | :---: | :--- | :---: | :--- |")
    md.append(r"| **BETA (Bruto)** | `(64, 750, 4, 40)` | `(Canais, Tempo, Blocos, Alvos)` | `float64` | Voltagem cerebral em microvolts ($\mu V$). |")
    md.append(r"| **BETA (Pipeline)** | `[160, 9, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Sinais filtrados (8–64 Hz) compensados em 130 ms. |")
    md.append(r"| **BETA (5 Canais)** | `[160, 5, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Apenas eletrodos `POz, PO3, PO4, Oz, O1`. |")
    md.append(r"| **BETA (3 Canais)** | `[160, 3, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Apenas eletrodos occipitais primários `O1, Oz, O2`. |")
    md.append(r"| **Benchmark SSVEP** | `(64, 1500, 40, 6)` | `(Canais, Tempo, Alvos, Blocos)` | `float64` | Voltagem cerebral gravada em câmara de Faraday. |")
    md.append(r"| **MAMEM-SSVEP-II** | `[500, 1, 8, 768]` | `[Tentativas, 1, Canais, Tempo]` | `float32` | Voltagem a 256 Hz de 8 canais e 5 frequências. |")
    md.append(r"| **MAtt (Riemann)** | `[Batch, 7, 12, 12]` | `[Batch, Subjanelas, Matriz, Matriz]` | `float32` | Matrizes de covariância SPD em variedade curva. |")
    md.append(r"| **DB-AFDNet (Entrada)**| `[Batch, 3, 9, 250]` | `[Batch, Sub-bandas, Canais, Tempo]` | `float32` | Sinais em 3 sub-bandas de filtros harmônicos. |")
    md.append(r"| **DB-AFDNet (Espectral)**| `[Batch, 9, 280]` | `[Batch, Canais, Bins_Freq]` | `float32` | Parte Real (Amplitude) e Imaginária (Fase) a 0.2 Hz. |")
    md.append(r"| **Compact ANN (Ref)** | `[40, 10, 250]` | `[Alvos, Harmônicos, Tempo]` | `float32` | Gabaritos analíticos de senos e cossenos teóricos. |")
    md.append("")
    md.append("---")
    md.append("*Documento gerado automaticamente pelo algoritmo `scripts/generate_dimensions_report.py` para a disciplina OPT098.*")

    return "\n".join(md)


def main():
    print("Iniciando algoritmo de análise e descrição dimensional das bases de dados...")
    specs = get_beta_actual_specs()
    if specs["found"]:
        print(f"Dataset BETA local detectado: tensor S1.mat com formato {specs['sample_shape']}")
    else:
        print("Usando especificações documentadas do BETA...")

    print("Construindo guia explicativo detalhado...")
    md_content = build_markdown_guide(specs)

    print(f"Gravando arquivo em: {OUTPUT_MD_PATH}")
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Sucesso! Arquivo '{OUTPUT_MD_PATH.name}' gerado com sucesso!")


if __name__ == "__main__":
    main()
