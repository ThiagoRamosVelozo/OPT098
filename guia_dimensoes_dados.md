# Guia Explicativo das Dimensões dos Dados e Benchmarks (SSVEP-BCI)

> **Objetivo deste Documento:** Explicar detalhadamente, até para quem não tem formação técnica prévia em ciência de dados ou neuroengenharia, o que é cada uma das **dimensões** presentes em cada uma das bases de dados (*benchmarks*) e modelos deste projeto, de onde vêm esses números, do que cada dimensão é feita fisicamente e como elas se transformam ao longo do pipeline de processamento.

---

## 1. O que é uma 'Dimensão' em Dados de Eletroencefalograma? (Para Leigos)

Imagine que você queira organizar informações em uma planilha:

* **1 Dimensão (1D - Uma Linha / Vetor):** É uma fita contínua de medidas. Exemplo: a voltagem elétrica gravada por um único sensor segundo a segundo: `[1.2, 1.5, 0.8, -0.4, ...]`. Tem apenas comprimento: `(Tempo)`.
* **2 Dimensões (2D - Uma Tabela / Matriz):** É uma folha do Excel com linhas e colunas. Cada linha representa um sensor na cabeça e cada coluna representa um instante no tempo: `(Sensores, Tempo)`.
* **3 Dimensões (3D - Um Caderno / Cubo):** Várias folhas empilhadas. Se o voluntário repetiu o teste 4 vezes seguidas, temos uma folha 2D para cada repetição (bloco): `(Sensores, Tempo, Repetições)`.
* **4 Dimensões (4D - Uma Estante de Cadernos / Hipercubo):** Um conjunto de cubos. Se existem 40 letras diferentes na tela do computador para o usuário escolher com o olhar, temos um caderno inteiro para cada uma das 40 letras: `(Sensores, Tempo, Repetições, Letras)`.

Cada dimensão responde a uma pergunta específica:
1. **Onde** no cérebro medimos? $\rightarrow$ *Dimensão de Canais/Eletrodos*.
2. **Quando** aconteceu a medida? $\rightarrow$ *Dimensão de Tempo/Amostras*.
3. **Quantas vezes** repetimos? $\rightarrow$ *Dimensão de Blocos/Tentativas*.
4. **O que** o usuário estava tentando digitar/olhar? $\rightarrow$ *Dimensão de Alvos/Classes*.
5. **Em qual frequência harmônica** estamos analisando? $\rightarrow$ *Dimensão de Sub-bandas*.

---

## 2. Benchmark 1: Dataset BETA (Liu et al., 2020) — Nossa Base Principal

O **BETA** (*Benchmark database toward SSVEP-BCI Application*) é a base pública com dados reais gravados de 70 voluntários sob um teclado visual de soletrar com 40 teclas cintilantes.

### 2.1. O Tensor Bruto Original no MATLAB: `(64, 750, 4, 40)`

Ao carregar um arquivo como `S1.mat` em Python (`scipy.io.loadmat`), a variável `data.EEG` possui exatamente quatro eixos:

```text
                          EIXO 3: 4 Blocos (Repetições)
                                /
                               /
              ┌───────────────────────────┐
              │                           ││
              │                           ││
  EIXO 1:     │                           ││   EIXO 4: 40 Alvos/Teclas
  64 Eletrodos│        SINAIS DE EEG      ││  (8.0 Hz a 15.8 Hz)
  (FP1 a VEO) │                           ││
              │                           ││
              │                           ││
              └───────────────────────────┘/
                ─────────────────────────
                EIXO 2: 750 Pontos no Tempo
                   (3.0 segundos a 250 Hz)
```

#### Dissecando cada uma das 4 dimensões do BETA:

### **Dimensão 1: 64 Canais (Eletrodos no Escalpo)**
* **O que é fisicamente:** São 64 pequenos pinos condutores de metal posicionados sobre o couro cabeludo do voluntário com gel condutor, organizados de acordo com o padrão internacional **10-20**.
* **Unidade de medida:** Microvolts ($\mu V$), ou seja, milionésimos de volt ($10^{-6}\text{ V}$). Para os sujeitos do BETA, essa voltagem oscila tipicamente entre $-500\mu V$ e $+500\mu V$.
* **Quais são os eletrodos:**
  * 62 eletrodos no cérebro: da testa (`FP1, FPz, FP2`), centro da cabeça (`Cz, C3, C4`) até a nuca (`Oz, O1, O2`).
  * 2 eletrodos de eletrooculograma (EOG): `HEO` (olho horizontal) e `VEO` (olho vertical), usados para monitorar piscadas e desvio ocular.
* **Os 9 canais parieto-occipitais chave:** O córtex visual humano fica na nuca. O cérebro só processa o SSVEP na região occipital posterior. Por isso, a literatura seleciona apenas 9 canais: `[Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2]` (índices 47, 53, 54, 55, 56, 57, 60, 61, 62).

### **Dimensão 2: 750 Pontos Temporais (Amostras)**
* **O que é fisicamente:** É a passagem do tempo discretizada pelo conversor analógico-digital (amplificador SynAmps2 da Neuroscan).
* **Taxa de amostragem ($f_s$):** $250\text{ Hz}$, significando que o aparelho mede a voltagem **250 vezes por segundo** ($\Delta t = \frac{1}{250} = 4\text{ milissegundos}$ entre cada ponto).
* **Duração total:** $\frac{750\text{ amostras}}{250\text{ Hz}} = 3,0\text{ segundos}$.
* **Divisão interna desses 3 segundos (Linha do Tempo de uma Tentativa):**
  1. **Amostras 1 a 125 (0,0s a 0,5s — 125 amostras):** Período de pré-estímulo / fixação. Uma cruz vermelha aparece na tela indicando onde o voluntário deve olhar, mas os quadrados ainda não estão piscando.
  2. **Amostras 126 a 157 (0,50s a 0,63s — 32 amostras = 130 ms):** Latência visual. A luz começa a piscar, mas o sinal luminoso leva 130 milissegundos para viajar da retina pelo nervo óptico até atingir os neurônios do córtex visual na nuca. **Esses 130 ms são descartados nos modelos!**
  3. **Amostras 158 a 625 (0,63s a 2,5s — 500 amostras = 2,0 s):** Estímulo visual ativo puro. Os neurônios da nuca disparam em regime estacionário.
  4. **Amostras 626 a 750 (2,5s a 3,0s — 125 amostras = 0,5 s):** Pós-estímulo / descanso para o voluntário desviar o olhar.

### **Dimensão 3: 4 Blocos (Sessões / Repetições)**
* **O que é fisicamente:** O voluntário não faz o teste apenas uma vez. Para garantir robustez e evitar que um espirro ou piscada estrague os dados, a pessoa digita todas as 40 teclas do teclado 4 vezes separadas (Bloco 1, Bloco 2, Bloco 3 e Bloco 4).
* **Intervalo entre blocos:** O voluntário descansa alguns minutos entre cada bloco para relaxar os olhos e não sofrer com cansaço visual.

### **Dimensão 4: 40 Alvos (Classes de Estímulo / Teclas Visuais)**
* **O que é fisicamente:** São os 40 caracteres do teclado virtual na tela (letras de A a Z, números de 0 a 9 e caracteres especiais).
* **Frequências de cintilação:** Variam de **8,0 Hz a 15,8 Hz** com passos regulares de **0,2 Hz** (ex.: $8,0\text{ Hz}, 8,2\text{ Hz}, 8,4\text{ Hz}, \dots, 15,8\text{ Hz}$).
* **Modulação conjunta de fase (JFPM):** Como 0,2 Hz é uma diferença muito pequena para o olho humano separar puramente pela frequência, o BETA codifica uma fase inicial diferente em cada tecla: $0, 0.5\pi, 1.0\pi, 1.5\pi$ radianos ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).

---

### 2.2. O Tensor Reorganizado para Treinamento (Issue 03): `[160, Canais, Tempo]`

Para treinar modelos clássicos e neurais, unificamos os blocos e alvos na dimensão de **amostras de treino (exemplos)**:

$$\text{Total de Tentativas por Sujeito} = 4\text{ blocos} \times 40\text{ alvos} = 160\text{ amostras}$$

O tensor final que o módulo de dados (`src/data/loader.py`) entrega possui formato:

| Cenário Experimental | Dimensões do Tensor `[N, C, T]` | Explicação dos Eixos |
| :--- | :---: | :--- |
| **9 Canais (1.0 s)** | `[160, 9, 250]` | 160 tentativas, 9 eletrodos parieto-occipitais, 250 amostras (1 segundo a 250 Hz). |
| **5 Canais (1.0 s)** | `[160, 5, 250]` | 160 tentativas, 5 eletrodos centrais (`POz, PO3, PO4, Oz, O1`), 250 amostras. |
| **3 Canais (1.0 s)** | `[160, 3, 250]` | 160 tentativas, 3 eletrodos mínimos (`O1, Oz, O2`), 250 amostras. |
| **9 Canais (0.5 s)** | `[160, 9, 125]` | 160 tentativas, 9 eletrodos, 125 amostras temporais (alta velocidade de comando). |
| **9 Canais (0.3 s)** | `[160, 9, 75]`  | 160 tentativas, 9 eletrodos, 75 amostras temporais (janela ultracurta). |

---

## 3. Benchmark 2: Benchmark Dataset (Wang et al., 2016)

O **Benchmark Dataset** da Universidade de Tsinghua é o conjunto de dados mais citado da história de SSVEP, gravado com 35 voluntários dentro de uma câmara de blindagem eletromagnética estrita (Gaiola de Faraday).

### 3.1. O Tensor no MATLAB: `(64, 1500, 40, 6)`

Observe a sutil inversão dos eixos 3 e 4 em relação ao BETA:

| Eixo | Tamanho | O que representa fisicamente | Comparação com o BETA |
| :-: | :-: | :--- | :--- |
| **1** | **64** | Eletrodos cerebrais (sistema 10-20). | Idêntico ao BETA (mesma touca Neuroscan). |
| **2** | **1500** | Pontos temporais ($6,0$ segundos a $250\text{ Hz}$). | **O dobro do BETA** (5s de estimulação vs 2s do BETA). |
| **3** | **40** | Alvos visuais (8,0 a 15,8 Hz em passos de 0,2 Hz). | Idênticas frequências e fases JFPM. |
| **4** | **6** | Blocos de repetição por voluntário. | **6 repetições** (BETA tem apenas 4). |

> **Atenção Técnica no Código:** No Benchmark, a chave do MATLAB é `mat['data']` com formato `(64, 1500, 40, 6)`, enquanto no BETA é `mat['data']['EEG'][0][0]` com formato `(64, 750, 4, 40)`. Essa diferença exige a transposição `transpose(2, 3, 0, 1)` no Benchmark e `transpose(3, 2, 0, 1)` no BETA, exatamente como documentado no `references/DBAFDNet/main.py#L65-L72`.

---

## 4. Benchmark 3: MAMEM-SSVEP-II Dataset (Pan et al., 2022 / MAtt)

Utilizado no artigo do **MAtt** (*NeurIPS 2022*), este dataset foca em decodificação geométrica sobre variedades não-euclidianas.

### 4.1. Dimensões no Domínio do Tempo Euclidiano: `[Batch, 1, 8, 768]`
* **1 (Canal de Imagem/Batch):** Representa o canal único de entrada para convoluções 2D no PyTorch.
* **8 Canais Focais:** Apenas 8 eletrodos na nuca: `[PO7, PO3, POz, PO4, PO8, O1, Oz, O2]`.
* **768 Pontos Temporais:** Amostrados a $256\text{ Hz}$ durante $3,0$ segundos ($3 \times 256 = 768$ amostras).
* **5 Classes:** Frequências mais espaçadas e lentas: $6.66\text{ Hz}, 7.50\text{ Hz}, 8.57\text{ Hz}, 10.00\text{ Hz}, 12.00\text{ Hz}$.

### 4.2. A Transformação Riemanniana `E2R` (Euclidiano para Matriz SPD): `[Batch, M, 12, 12]`
O MAtt substitui a leitura de voltagem direta por **matrizes de covariância**: 

1. **$M$ Sub-janelas Temporais (`epochs = 7`):** O sinal de 3 segundos é cortado em 7 pedacinhos temporais consecutivos.
2. **$12 \times 12$ Matriz SPD (Simétrica Definida Positiva):** Cada pedacinho temporal gera uma matriz $12 \times 12$ que mede como cada canal elétrico se correlaciona com todos os outros:
   $$C = \frac{(X - \bar{X})(X - \bar{X})^T}{T - 1} + \epsilon I$$
3. Essa matriz vive em um espaço curvo (variedade Riemanniana), onde o MAtt calcula distâncias geodésicas (distância sobre a curvatura da esfera cerebral) em vez da distância em linha reta euclidiana.

---

## 5. Estrutura de Tensores do DB-AFDNet (Yang et al., 2026)

O DB-AFDNet processa o sinal através de uma esteira complexa de álgebra linear e decomposição espectral.

```text
  ENTRADA TEMPORAL             BANCO DE FILTROS           DOMÍNIO ESPECTRAL COMPLEXO (RFFT)
  [Batch, 9, 250]     ───►    [Batch, 3, 9, 250]   ───►   Parte Real:  [Batch, 9, 280]
  (Sinal no tempo)            (3 sub-bandas)              Parte Imag:  [Batch, 9, 280]
                                                          (280 frequências de 8 a 64 Hz)
```

### 5.1. O que é o Tensor `[Batch, 3, 9, 250]` (Entrada da Rede)?
* **Batch:** Quantidade de épocas processadas de uma vez na GPU (ex.: 32 ou 64 tentativas).
* **3 Sub-bandas (Filter Bank):** O sinal original é filtrado em três faixas de frequência sobrepostas:
  * **Sub-banda 1 (8 a 64 Hz):** Captura o sinal SSVEP fundamental e todas as suas harmônicas.
  * **Sub-banda 2 (16 a 64 Hz):** Remove o ritmo alfa espontâneo (8-12 Hz) que atrapalha e isola o 2º harmônico.
  * **Sub-banda 3 (24 a 64 Hz):** Isola os harmônicos de alta ordem (3º e 4º harmônicos).
* **9 Canais:** Os eletrodos parieto-occipitais ativos.
* **250 Pontos:** 1,0 segundo de estimulação.

### 5.2. O que é o Tensor `[Batch, 9, 280]` (Domínio da Frequência)?
* **Por que 280 pontos se a entrada tinha 250 pontos?**
  * A rede aplica `torch.fft.rfft(x, n=1250, axis=-1)`. Ao preencher o sinal com zeros (*zero-padding*) até 1250 pontos, a resolução da Transformada de Fourier torna-se ultra-fina:
    $$\Delta f = \frac{f_s}{N_{\text{fft}}} = \frac{250\text{ Hz}}{1250} = 0,20\text{ Hz por ponto}$$
  * Isso é perfeito, pois os alvos do BETA são separados exatamente por $0,20\text{ Hz}$!
* **O que são os índices 40 a 320 (`40:320` $\rightarrow$ 280 bins):**
  * O bin 40 corresponde a: $40 \times 0,20\text{ Hz} = 8,0\text{ Hz}$ (menor frequência do teste).
  * O bin 320 corresponde a: $320 \times 0,20\text{ Hz} = 64,0\text{ Hz}$ (maior frequência do teste).
  * Total: $320 - 40 = 280$ pontos no eixo de frequência.
* **A Divisão em Dois Ramos:**
  * **Ramo Real (Amplitude / Cosseno):** Mede a energia e potência de cada onda.
  * **Ramo Imaginário (Fase / Seno):** Mede o atraso e a sincronia temporal das ondas.

---

## 6. Estrutura de Tensores da Compact ANN (Wang et al., 2023)

A **Compact ANN** reduz os parâmetros treináveis para ínfimos 335 pesos usando matrizes fixas de gabarito:

### 6.1. Matriz de Referência Senoidal Teórica $R_i$: `[10, 250]`
* Para cada um dos 40 alvos $i$, gera-se uma matriz fixa contendo senos e cossenos da frequência fundamental $f_i$ e de 4 harmônicos ($2f_i, 3f_i, 4f_i, 5f_i$):
* $5\text{ harmônicos} \times 2\text{ componentes (seno e cosseno)} = 10\text{ linhas}$.
* 250 colunas correspondentes aos instantes de tempo.

### 6.2. Gabarito Empírico Médio $\bar{X}_i$: `[9, 250]`
* É a média de voltagem calculada através dos blocos anteriores do próprio sujeito, servindo como uma 'impressão digital cerebral' de como aquele voluntário específico reage àquela tecla.

---

## 7. Tabela Comparativa Mestra de Todas as Dimensões

| Base / Modelo | Formato do Tensor | Dimensões Detalhadas | Tipo de Dado | O que cada elemento guarda fisicamente |
| :--- | :---: | :--- | :---: | :--- |
| **BETA (Bruto)** | `(64, 750, 4, 40)` | `(Canais, Tempo, Blocos, Alvos)` | `float64` | Voltagem cerebral em microvolts ($\mu V$). |
| **BETA (Pipeline)** | `[160, 9, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Sinais filtrados (8–64 Hz) compensados em 130 ms. |
| **BETA (5 Canais)** | `[160, 5, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Apenas eletrodos `POz, PO3, PO4, Oz, O1`. |
| **BETA (3 Canais)** | `[160, 3, 250]` | `[Tentativas, Canais, Tempo]` | `float32` | Apenas eletrodos occipitais primários `O1, Oz, O2`. |
| **Benchmark SSVEP** | `(64, 1500, 40, 6)` | `(Canais, Tempo, Alvos, Blocos)` | `float64` | Voltagem cerebral gravada em câmara de Faraday. |
| **MAMEM-SSVEP-II** | `[500, 1, 8, 768]` | `[Tentativas, 1, Canais, Tempo]` | `float32` | Voltagem a 256 Hz de 8 canais e 5 frequências. |
| **MAtt (Riemann)** | `[Batch, 7, 12, 12]` | `[Batch, Subjanelas, Matriz, Matriz]` | `float32` | Matrizes de covariância SPD em variedade curva. |
| **DB-AFDNet (Entrada)**| `[Batch, 3, 9, 250]` | `[Batch, Sub-bandas, Canais, Tempo]` | `float32` | Sinais em 3 sub-bandas de filtros harmônicos. |
| **DB-AFDNet (Espectral)**| `[Batch, 9, 280]` | `[Batch, Canais, Bins_Freq]` | `float32` | Parte Real (Amplitude) e Imaginária (Fase) a 0.2 Hz. |
| **Compact ANN (Ref)** | `[40, 10, 250]` | `[Alvos, Harmônicos, Tempo]` | `float32` | Gabaritos analíticos de senos e cossenos teóricos. |

---
*Documento gerado automaticamente pelo algoritmo `scripts/generate_dimensions_report.py` para a disciplina OPT098.*