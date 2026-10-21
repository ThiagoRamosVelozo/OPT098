Para garantir que o projeto seja entregue com rigor científico dentro dos próximos dois meses sem sobrecarregar ninguém, a melhor estratégia é dividi-lo em **etapas modulares com entregas parciais a cada 2 semanas**.

Como a base teórica e o código de partida já estão consolidados com os artigos analisados (especialmente o **DB-AFDNet**, a **Compact ANN** e a base **BETA**), vocês não vão começar do zero. Vocês vão montar um pipeline experimental comparativo em Python e PyTorch.

Abaixo está o plano operacional completo: **O que fazer**, **Como fazer**, **A divisão de tarefas entre você (Victor) e o Thiago** e o **Cronograma prático**.

---

### 1. O que vamos fazer (O Projeto)

* **Objetivo Geral:** Desenvolver e avaliar um pipeline de decodificação de SSVEP inter-sujeito (*cross-subject*, sem calibração prévia) utilizando a base de dados pública **BETA**.


* **Perguntas de Pesquisa que o projeto vai responder:**
1. *Impacto da redução de canais:* O modelo consegue manter acurácia e taxa de transferência de informação (ITR) viáveis utilizando apenas 5 canais (POz, PO3, PO4, Oz, O1) ou 3 canais (O1, Oz, O2) em vez dos 9 eletrodos parieto-occipitais padrão?


2. *Impacto de janelas ultracurtas:* Como o algoritmo se comporta com janelas de tempo de 0,3 s, 0,5 s e 0,8 s (alta velocidade de comando) frente à janela clássica de 1,0 s?


3. *Linha de base comparativa:* Como a abordagem neural com mecanismos de atenção se compara ao método clássico de correlação canônica não supervisionado (FBCCA) sob essas mesmas restrições de canais e tempo?





---

### 2. Como vamos fazer (Abordagem Técnica e Ferramental)

* **Linguagem e Bibliotecas:** Python 3.10+, PyTorch, NumPy, SciPy (para manipulação dos arquivos `.mat`), MNE-Python (para manipulação e filtragem de EEG) e Matplotlib/Seaborn para os gráficos de resultado.
* **Dados:** Download de um subconjunto estruturado do dataset **BETA** (70 voluntários, 40 alvos, 250 Hz).


* **Modelos Computacionais:**
* *Baseline Clássico:* Algoritmo **FBCCA** (banco de filtros harmônicos com CCA), que não requer treino e serve de chão comparativo.


* *Modelo Neural Principal:* Repositório oficial do **DB-AFDNet**. O código já está estruturado em PyTorch com rotina de validação *Leave-One-Subject-Out* (LOSO).


* *Adaptação/Compactação:* Parametrização e ajuste fino dos módulos convolucionais e de atenção (DIAN) para operar diretamente com vetores de 3, 5 e 9 canais.





---

### 3. Divisão de Papéis e Tarefas (Quem faz o quê e como)

A divisão foi pensada para equilibrar a carga de programação, processamento computacional e escrita técnica:

#### **Victor Angelo (Foco: Pipeline de Dados, Baselines e Engenharia de Canais)**

* **Responsabilidade Técnica:**
* **Ingestão e Pré-processamento:** Criar o script de carregamento dos arquivos `.mat` do dataset BETA, aplicando o corte de latência (130 ms) e os filtros passa-faixa (8–64 Hz).


* **Módulo de Seleção Espacial:** Implementar a lógica que permite selecionar os subconjuntos de eletrodos (9 canais $\rightarrow$ 5 canais $\rightarrow$ 3 canais) de forma configurável via parâmetros de linha de comando ou arquivo `.yaml`.


* **Implementação do Baseline FBCCA:** Estruturar o classificador clássico FBCCA (usando bancos de senoides de referência teórica) para rodar exatamente sob as mesmas fatias de dados do modelo neural.


* **Métricas e Avaliação:** Implementar a função de cálculo da Taxa de Transferência de Informação (ITR em bits/min) e acurácia.





#### **Thiago Ramos (Foco: Arquitetura Neural, Treinamento/LOSO e Ablação)**

* **Responsabilidade Técnica:**
* **Ambiente e Repositório Neural:** Clonar, configurar e rodar o repositório do **DB-AFDNet** no ambiente da máquina/GPU.


* **Adaptação das Camadas do Modelo:** Ajustar as camadas de entrada da rede (`Channel-wise Attention` e `Conv Block 1` do DIAN) para aceitar tensores com dimensões variáveis ($C=3, 5, 9$) sem quebrar a retropropagação.


* **Loop de Treinamento Cross-Subject (LOSO):** Configurar o particionamento *Leave-One-Subject-Out*, gerenciando o treino sobre $N-1$ sujeitos e teste no sujeito restante.


* **Exportação dos Resultados e Curvas:** Estruturar a geração de logs e matrizes de confusão/t-SNE para posterior análise estatística.





#### **Trabalho Conjunto (Victor + Thiago):**

* Execução dos testes em lote nas máquinas.
* Discussão dos resultados e confecção das figuras comparativas (gráficos de barra de ITR vs. número de canais).
* Redação do artigo final da disciplina.

---

### 4. Cronograma de Execução (8 Semanas)

| Período | Meta / Entrega | Ações Práticas |
| --- | --- | --- |
| **Semanas 1 e 2** | **Setup e Baseline** | • Download de dados do BETA (ex.: primeiros 10 a 15 sujeitos para prototipação rápida).

<br>

<br>• Victor monta o script de carregamento e o baseline FBCCA.

<br>

<br>• Thiago coloca o código do DB-AFDNet para rodar uma época com sucesso.

 |
| **Semanas 3 e 4** | **Experimento 1: Redução de Canais** | • Rodar o DB-AFDNet e o FBCCA avaliando a acurácia com **9 canais**, **5 canais** e **3 canais** (janela fixa de 1,0 s).

<br>

<br>• Salvar métricas parciais e gerar os primeiros gráficos comparativos. |
| **Semanas 5 e 6** | **Experimento 2: Janelas Ultracurtas** | • Rodar a melhor configuração de canais variando as janelas temporais: **0,3 s, 0,5 s, 0,8 s e 1,0 s**.

<br>

<br>• Victor computa as curvas de ITR (bits/min) e Thiago compila a tabela de acurácia média e desvio padrão.

 |
| **Semanas 7 e 8** | **Consolidação e Relatório Final** | • Comparação estatística (testes $t$ pareados entre FBCCA e modelo neural).

<br>

<br>• Redação conjunta do artigo/relatório final seguindo o template SBC com os gráficos prontos. |