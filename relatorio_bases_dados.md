# Relatório Explicativo das Bases de Dados e Modelagem de Sinais (OPT098)

> **Objetivo deste Relatório:** Apresentar uma visão analítica, completa e didática sobre as bases de dados de Eletroencefalograma (EEG), os dados do repositório neural **DB-AFDNet** e as Interfaces Cérebro-Computador (BCI) utilizadas neste projeto. Este documento foi elaborado com explicações intuitivas para que qualquer pessoa leiga compreenda perfeitamente o significado de cada métrica, eletrodo, transformação espectral e protocolo experimental.

---

## 1. Glossário Essencial: O que é tudo isso? (Para Leigos)

Para entender os números e tabelas a seguir, aqui está uma explicação direta dos conceitos da neurociência computacional:

1. **O que é uma Interface Cérebro-Computador (BCI)?**
   * É um sistema de hardware e software que lê as ondas elétricas do cérebro em tempo real e as traduz diretamente em comandos em um computador (como escolher letras em um teclado virtual ou mover um cursor), sem que a pessoa precise mexer nenhum músculo ou falar.

2. **O que é o sinal SSVEP (*Steady-State Visual Evoked Potential*)?**
   * Quando os nossos olhos olham para uma luz ou quadrado na tela que pisca em um ritmo fixo (por exemplo, 10 vezes por segundo, ou 10 Hz), os neurônios do córtex visual (na nuca) começam a disparar eletricidade exatamente nessa mesma frequência e em seus múltiplos (20 Hz, 30 Hz). O computador 'escuta' o cérebro, detecta a frequência que está brilhando mais forte e sabe para qual letra a pessoa está olhando.

3. **O que são Eletrodos e Canais?**
   * São pequenos sensores de metal encostados no couro cabeludo do voluntário. Cada eletrodo funciona como um 'microfone' biológico que grava a voltagem elétrica produzida pelos neurônios. Quanto mais longe do cérebro visual (olhos e testa), mais ruído e menos sinal útil existe.

4. **Por que focar nos Canais Parieto-Occipitais (Atrás da Cabeça)?**
   * O córtex visual humano fica localizado no osso occipital (nuca). Eletrodos como **Oz, O1, O2, POz, PO3, PO4, Pz** ficam colados no centro do processamento da visão. O restante dos 64 eletrodos espalhados pela cabeça capta muitos sinais que não têm relação com a visão (como pensamentos, movimentos dos dentes ou piscadas de olho).

5. **O que é Relação Sinal-Ruído (SNR)?**
   * Mede o quão 'forte' é a resposta visual em comparação ao ruído ambiente e biológico do cérebro. Como a voltagem cerebral é minúscula (na casa de microvolts, $\mu V$), o SNR em EEG é sempre negativo em decibéis (dB), o que significa que o ruído natural é maior do que o sinal que queremos detectar. Quanto mais perto de zero (menos negativo), melhor e mais nítido é o sinal.

6. **O que é Taxa de Transferência de Informação (ITR em bits/min)?**
   * É o 'velocímetro' de uma BCI. Representa quantos bits de informação limpa a pessoa consegue transmitir por minuto. Um sistema com 95% de acurácia que leva 5 segundos para decidir cada letra é **muito mais lento** (menor ITR) do que um sistema com 80% de acurácia que decide em apenas 0,5 segundo!

7. **O que é Latência Visual (130 ms)?**
   * Quando a luz pisca na tela, a luz viaja até a retina, é convertida em impulsos elétricos pelo nervo óptico e viaja até a nuca. Esse caminho leva em média **130 milissegundos (0,13 segundo)**. Por isso, os primeiros 130 ms após o início do estímulo são descartados, pois o cérebro ainda não começou a reagir.

---

## 2. As Bases de Dados Presentes e Referenciadas no Projeto

O projeto combina dados reais locais com referências bibliográficas do estado da arte:

### Tabela Comparativa Geral das Bases de Dados

| Característica / Base | **BETA Dataset** *(Base Local Principal)* | **Benchmark SSVEP** *(Wang et al., 2016)* | **MAMEM-SSVEP-II** *(Pan et al., 2022)* |
| :--- | :--- | :--- | :--- |
| **Artigo de Referência** | Liu et al. (2020), *Frontiers in Neuroscience* | Wang et al. (2016), *IEEE Trans. Biomed. Eng.* | Pan et al. (2022 - MAtt), *NeurIPS* |
| **Disponibilidade Local** | **Sim** (Arquivos `S1.mat` a `S10.mat` em `data/raw/`) | *Referência externa* (DB-AFDNet / Compact ANN) | *Referência externa* (Repositório MAtt) |
| **Número de Voluntários** | **70 participantes** (10 baixados para prototipação) | 35 participantes | 11 participantes |
| **Total de Eletrodos** | **64 eletrodos** (sistema 10-20 internacional) | 64 eletrodos | 8 eletrodos focais |
| **Eletrodos Foco do Projeto** | **9, 5 e 3 canais occipitais** | 9 canais parieto-occipitais | 8 canais occipitais |
| **Taxa de Amostragem ($f_s$)** | **250 Hz** (250 amostras por segundo) | 250 Hz | 256 Hz |
| **Número de Classes / Alvos** | **40 alvos visuais** (teclado virtual completo) | 40 alvos visuais | 5 comandos visuais |
| **Faixa de Frequências** | **8.0 Hz a 15.8 Hz** (espaçamento de 0.2 Hz) | 8.0 Hz a 15.8 Hz (passo de 0.2 Hz) | 6.66 Hz, 7.50 Hz, 8.57 Hz, 10.0 Hz, 12.0 Hz |
| **Modulação de Fase** | **4 fases** ($0, 0.5\pi, 1.0\pi, 1.5\pi$) | 4 fases | Sem modulação de fase explícita |
| **Ambiente de Gravação** | **Mundo Real** (sem blindagem eletromagnética) | Laboratório blindado (Faraday Cage) | Laboratório controlado |
| **Duração da Janela** | 0.5s pré + 2.0s estímulo + 0.5s pós (3.0s total) | 0.5s pré + 5.0s estímulo (5.5s total) | 3.0s de estimulação contínua |
| **Compensação de Latência** | **130 ms** (atraso do nervo óptico) | 140 ms / 120 ms | Não especificada |
| **Nível de Desafio (SNR)** | **Alto** (SNR média baixa: -13.78 dB) | Médio (Sinais mais limpos: -10.92 dB) | Médio |

---

## 3. Radiografia Empírica dos Dados Baixados no Repositório (`data/raw/`)

O script inspecionou diretamente os **10 arquivos `.mat`** locais baixados do Figshare. Cada arquivo contém a estrutura matricial bruta de um participante:

* **Total de Participantes Inspecionados:** 10 voluntários (`S1` a `S10`) 
* **Idade Média:** 24.9 anos
* **Distribuição por Gênero:** 5 mulheres e 5 homens (amostragem balanceada)
* **Relação Sinal-Ruído Média (Wide-band SNR):** -14.59 dB (mostra o desafio de trabalhar com dados reais)
* **BCI Quotient Médio:** 94.7 pontos (índice fisiológico populacional de resposta visual)

### Tabela Individualizada por Participante

| Arquivo | Sujeito | Idade | Gênero | BCI Quotient | Wide SNR (dB) | Dimensões do Tensor Bruto `(C, T, B, A)` | Faixa de Voltagem ($\mu V$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S1.mat` | **S1** | 26.5 | female | 108.4 | -12.48 dB | `(64, 750, 4, 40)` | [-566.2, 454.8] |
| `S2.mat` | **S2** | 17.4 | male | 92.1 | -15.00 dB | `(64, 750, 4, 40)` | [-123.2, 88.2] |
| `S3.mat` | **S3** | 27.8 | female | 111.2 | -12.06 dB | `(64, 750, 4, 40)` | [-125.1, 130.2] |
| `S4.mat` | **S4** | 35.5 | male | 87.1 | -15.77 dB | `(64, 750, 4, 40)` | [-559.0, 436.2] |
| `S5.mat` | **S5** | 26.4 | male | 80.6 | -16.77 dB | `(64, 750, 4, 40)` | [-145.9, 121.3] |
| `S6.mat` | **S6** | 23.0 | female | 87.3 | -15.74 dB | `(64, 750, 4, 40)` | [-107.9, 122.9] |
| `S7.mat` | **S7** | 22.5 | male | 83.0 | -16.40 dB | `(64, 750, 4, 40)` | [-82.0, 111.5] |
| `S8.mat` | **S8** | 28.0 | male | 96.5 | -14.33 dB | `(64, 750, 4, 40)` | [-861.3, 1920.3] |
| `S9.mat` | **S9** | 21.3 | female | 102.9 | -13.33 dB | `(64, 750, 4, 40)` | [-1704.3, 1770.2] |
| `S10.mat` | **S10** | 20.1 | female | 98.4 | -14.03 dB | `(64, 750, 4, 40)` | [-1494.0, 1987.7] |

> **Entendendo as Dimensões `(64, 750, 4, 40)`:**
> * **64:** Número de eletrodos gravados na cabeça.
> * **750:** Pontos no tempo. A 250 amostras por segundo, 750 pontos equivalem a exatamente 3,0 segundos (0,5s de descanso + 2,0s de luz piscando + 0,5s de término).
> * **4:** Blocos de repetição (cada letra/alvo foi testada 4 vezes pelo voluntário).
> * **40:** Quantidade de teclas/alvos visuais diferentes exibidos na tela.

---

## 4. Verificação e Diagnóstico do Repositório DB-AFDNet e Transformação de Dados

O **DB-AFDNet** (*Dual-Branch Attention-Based Frequency Domain Network*, Yang et al., 2026) é a principal arquitetura neural do projeto. Ele consome os dados brutos do dataset BETA através de uma cadeia rigorosa de pré-processamento e transformação espectral complexa:

### Pipeline de Ingestão do DB-AFDNet sobre o Dataset BETA

```text
  [Arquivo BETA S*.mat] (64 canais, 750 pontos)
           │
           ▼
  [1. Seleção dos 9 Canais] Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2 (índices 47, 53, 54, 55, 56, 57, 60, 61, 62)
           │
           ▼
  [2. Compensação de Latência] Início em t_delay = 157 amostras (0.5s pré + 130ms de atraso visual)
           │
           ▼
  [3. Banco de Filtros Harmônicos] Decomposição em sub-bandas (3 sub-bandas paralelas)
           │
           ▼
  [4. Tensor de Entrada na Rede] Shape: [Batch, 3 sub-bandas, 9 canais, 250 amostras]
           │
           ▼
  [5. RFFT com Zero-Padding] rfft(x, n=1250) -> Resolução espectral de 0.2 Hz por ponto
           │
           ▼
  [6. Janela Espectral 8 a 64 Hz] Fatiamento dos índices 40 a 320 -> 280 pontos de frequência
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
  [Parte REAL (Amplitude)]                  [Parte IMAGINÁRIA (Fase)]
  Módulo DIAN com Atenção Espacial          Módulo DIAN com Atenção Espacial
  (Channel-wise Attention + Convs)          (Channel-wise Attention + Convs)
           │                                         │
           └────────────────────┬────────────────────┘
                                ▼
  [Classificação + Restrições] L = L_ce + 0.2*L_d (similaridade) + 0.2*L_s (ortogonalidade)
```

### Resultados da Inspeção do Código em `references/DBAFDNet`:

* **Status do Repositório Local:** Clonado e Acessível
* **Parâmetros Treináveis da Rede Principal (`DBACFNet`):** **4.340.312 parâmetros**
* **Teste de Forward-Pass em PyTorch:** **SUCESSO (Saída: [2, 40] classes)**
* **Tensor de Entrada Esperado:** `(Batch, 3, 9, 250)`
* **Pontos de Resolução Espectral (FFT bins):** `280 bins` (abrangendo 8.0 Hz a 64.0 Hz)
* **Função de Perda Multi-Objetivo:** `Loss = L_ce + 0.2 * L_sim + 0.2 * L_ortho`
* **Canais Específicos Mapeados no `main.py`:** `[47, 53, 54, 55, 56, 57, 60, 61, 62]` (9 eletrodos)
* **Amostra de Corte Temporal (Latência):** `157` pontos

### Os 12 Modelos Disponíveis no Repositório do DB-AFDNet:

O repositório oficial do DB-AFDNet já contém implementação pronta de 12 modelos comparativos na pasta `models/`:

| # | Nome do Modelo | Arquivo | Categoria / Descrição |
| :-: | :--- | :--- | :--- |
| 1 | **DBACFNet** | `DBACFNet.py` | Duplo ramo espectral complexo com atenção adaptativa (modelo principal) |
| 2 | **FBDBACFNet** | `FBDBACFNet.py` | DB-AFDNet integrado com banco de filtros harmônicos multiescala |
| 3 | **FBCCA** | `FBCCA.py` | Filter Bank Canonical Correlation Analysis (baseline clássico sem treino) |
| 4 | **TRCA** | `TRCA.py` | Task-Related Component Analysis (análise de correlação intra-sujeito) |
| 5 | **CCA** | `CCA.py` | Análise de Correlação Canônica pura |
| 6 | **FBSSVEPFormer**| `FBSSVEPFormer.py` | Transformer adaptado para SSVEP com banco de filtros |
| 7 | **EEGNet** | `EEGNet.py` | Rede convolucional compacta padrão de neurociência (Lawhern et al.) |
| 8 | **ConvCA** | `ConvCA.py` | Rede convolucional inspirada em correlação canônica |
| 9 | **GuneyDNN** | `GuneyDNN.py` | Rede neural densa profunda pioneira em SSVEP |
| 10 | **CCNN** | `CCNN.py` | Complex Convolutional Neural Network |
| 11 | **MSI** | `MSI.py` | Magnitude Squared Coherence / Synchronization Index |
| 12 | **ShallowConvNet**| `ShallowConvNet.py` | Rede convolucional rasa para decodificação de ritmos oscilatórios |

---

## 5. Engenharia de Redução de Eletrodos (O que o projeto investiga)

Um capacete de 64 eletrodos com gel condutor é inviável no dia a dia: demora mais de 30 minutos para colocar, suja o cabelo e causa desconforto. Nosso projeto avalia como o algoritmo se comporta ao usarmos **apenas 5 ou 3 sensores**:

```text
                     [ NARIZ / FRENTE ]
                            FPz
                       F3   Fz   F4
                       C3   Cz   C4
                       P3   Pz   P4
             --------------------------------
             |      REGIÃO PARIETO-OCCIPITAL |
             |   PO5   PO3   POz   PO4   PO6 |  <- (9 canais padrão)
             |          O1    Oz    O2       |  <- (3 canais mínimos)
             --------------------------------
                     [ NUCA / VISÃO ]
```

### Mapeamento dos 3 Subconjuntos de Canais:

1. **Montagem Padrão Completa da Literatura (9 Canais):**
   * **Eletrodos:** `Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2`
   * **Índices no Array (0-indexado):** `[45, 51, 52, 53, 54, 55, 58, 59, 60]`
   * **Objetivo:** Padrão de referência (*gold standard*) para máxima cobertura do córtex visual.

2. **Montagem Reduzida (5 Canais):**
   * **Eletrodos:** `POz, PO3, PO4, Oz, O1`
   * **Índices no Array (0-indexado):** `[53, 52, 54, 59, 58]`
   * **Objetivo:** Redução de quase 50% dos fios sem perder a linha média parieto-occipital.

3. **Montagem Mínima Ultracompacta (3 Canais):**
   * **Eletrodos:** `O1, Oz, O2`
   * **Índices no Array (0-indexado):** `[58, 59, 60]`
   * **Objetivo:** Modelo viável para óculos ou tiaras comerciais leves (*wearables*).

---

## 6. Panorama de Desempenho dos Algoritmos na Literatura

Como os métodos clássicos e as redes neurais se comportam frente a essas bases de dados:

| Algoritmo / Modelo | Tipo de Abordagem | Dataset Avaliado | Acurácia Média | Taxa de Transferência (ITR) | Tempo de Inferência | Requer Treino Prévio? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FBCCA** (Chen et al., 2015) | Correlação Canônica com Banco de Filtros | BETA (1.0s) | ~72.0% a 78.5% | 98.79 ± 4.49 bpm (em 1.4s) | < 0.005 s | **Não** (Sem calibração) |
| **msTRCA** (Wong et al., 2020) | Análise de Correlação de Tarefa Individual | BETA (0.6s) | ~88.4% | 145.26 ± 8.15 bpm | < 0.008 s | **Sim** (Calibração individual) |
| **Compact ANN** (Wang et al., 2023) | Rede Neural com Task-Attention (335 parâmetros) | BETA (0.6s) | ~85.2% | 137.31 bpm | < 0.004 s | **Sim** (Supervisionado) |
| **DB-AFDNet** (Yang et al., 2026) | Duplo Ramo Espectral + Atenção DIAN | **BETA (1.0s)** | **74.33%** | **154.76 bpm (pico)** | **0.0062 s** | **Não para o usuário** (Treino Inter-Sujeito LOSO) |
| **DB-AFDNet** (Yang et al., 2026) | Duplo Ramo Espectral + Atenção DIAN | **Benchmark (1.0s)** | **84.02%** | **162.10 bpm** | **0.0062 s** | **Não para o usuário** (Treino Inter-Sujeito LOSO) |
| **MAtt** (Pan et al., 2022) | Aprendizado Profundo em Variedades Riemannianas | MAMEM-SSVEP-II | 65.50 ± 8.20% | N/D | ~0.012 s | **Sim** (Cross-session) |

> **Insight Crucial:** O DB-AFDNet atinge acurácia de **74.33% no BETA** e **84.02% no Benchmark**. A diferença de quase 10% decorre diretamente da natureza dos dados: o dataset BETA foi gravado fora de cabine acústica/eletromagnética, reproduzindo as dificuldades reais que encontramos na casa ou no hospital do paciente.

---

## 7. Conclusões e Recomendações para os Próximos Experimentos

1. **Validade e Integridade Comprovada:** Todos os arquivos de dados locais em `data/raw/` possuem dimensões consistentes `(64, 750, 4, 40)` a 250 Hz, permitindo extrair sem cortes qualquer janela de 0.2s até 2.0s.
2. **Compensação Temporal Obrigatória:** Todo pré-processamento deve pular os primeiros 130 ms após o início do estímulo para sincronizar exatamente com o pico de resposta neural do córtex visual.
3. **Redução sem Colapso:** Reduzir de 9 para 5 canais mantém a cobertura occipital central e costuma preservar até 90% da acurácia. A transição de 5 para 3 canais exige a atenção adaptativa multiescala da rede para compensar a perda de resolução espacial.
4. **Ponto Ótimo de Velocidade (ITR):** Embora janelas de 1.0s forneçam maior acurácia absoluta, as janelas de 0.5s e 0.6s costumam entregar a maior velocidade de digitação em palavras por minuto, o que será quantificado no relatório final da disciplina.

---
*Relatório gerado automaticamente por `scripts/generate_dataset_report.py` para a disciplina OPT098.*