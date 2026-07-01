# Fake News Detector

Classificador de notícias falsas utilizando **Machine Learning**, desenvolvido em **Python** com o algoritmo **Random Forest**. O projeto utiliza uma base de dados do Kaggle contendo notícias verdadeiras e falsas para treinar um modelo capaz de classificar novas notícias.

## Objetivo

Desenvolver um classificador de notícias falsas aplicando técnicas de Processamento de Linguagem Natural (NLP) e Aprendizado de Máquina.

O fluxo do projeto consiste em:

1. Carregar a base de dados.
2. Realizar o pré-processamento dos textos.
3. Converter os textos para representação numérica utilizando TF-IDF.
4. Treinar um classificador Random Forest.
5. Avaliar o desempenho do modelo.
6. Classificar novas notícias.

---

## Estrutura do Projeto

```
FakeNewsClassifier/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── train.py
├── predict.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

### Arquivos

| Arquivo | Descrição |
|----------|-----------|
| `train.py` | Responsável pelo treinamento e avaliação do modelo. |
| `predict.py` | Utilizado para classificar novas notícias utilizando o modelo treinado. |
| `model.pkl` | Modelo Random Forest treinado. |
| `vectorizer.pkl` | Vetorizador TF-IDF treinado. |
| `Fake.csv` | Notícias falsas. |
| `True.csv` | Notícias verdadeiras. |

---

## Base de Dados

O projeto utiliza a base de dados **Fake and Real News Dataset**, disponível no Kaggle.

A base contém dois arquivos:

- `Fake.csv`
- `True.csv`

Cada notícia possui informações como:

- título (`title`)
- texto (`text`)
- assunto (`subject`)
- data (`date`)

Durante o treinamento são criadas as seguintes classes:

| Classe | Significado |
|---------|-------------|
| 0 | Fake News |
| 1 | True News |

---

## Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib

---

## Algoritmo Utilizado

O algoritmo escolhido foi o **Random Forest**, um método de aprendizado supervisionado baseado em um conjunto de Árvores de Decisão.

A classificação é realizada através da votação entre diversas árvores, reduzindo o risco de overfitting e aumentando a capacidade de generalização do modelo.

---

## Pré-processamento

Antes do treinamento, os textos passam pelas seguintes etapas:

- Conversão para letras minúsculas;
- Remoção de URLs;
- Remoção de caracteres especiais;
- Remoção de números;
- Remoção de espaços extras;
- Vetorização utilizando TF-IDF.

---

## Representação dos Textos

Os textos são convertidos para vetores numéricos utilizando o algoritmo **TF-IDF (Term Frequency – Inverse Document Frequency)**.

Essa representação permite que o algoritmo Random Forest trabalhe com dados textuais.

---

## Divisão da Base

A base é dividida em:

- 80% para treinamento
- 20% para teste

A divisão é realizada utilizando `train_test_split()`.

---

## Métricas Avaliadas

O desempenho do modelo é avaliado utilizando:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusão

---

## Instalação

Clone o projeto:

```bash
git clone https://github.com/vinicitadin/fake_news_detector
```

Entre na pasta:

```bash
cd FakeNewsClassifier
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Treinamento

Execute:

```bash
python train.py
```

Ao final do treinamento serão gerados:

```
model.pkl
vectorizer.pkl
```

Também serão exibidas:

- Accuracy
- Classification Report
- Confusion Matrix

---

## Classificação de uma Nova Notícia

Após o treinamento execute:

```bash
python predict.py
```

Digite uma notícia quando solicitado.

Exemplo:

```
Digite a notícia:

Scientists discover a new vaccine capable of curing...
```

Saída:

```
True News
```

ou

```
Fake News
```

---

## Fluxo do Projeto

```
                Fake.csv
                True.csv
                    │
                    ▼
          Pré-processamento
                    │
                    ▼
               Vetorização
                 (TF-IDF)
                    │
                    ▼
            Random Forest
                    │
                    ▼
        Avaliação do Modelo
                    │
                    ▼
      model.pkl + vectorizer.pkl
                    │
                    ▼
          Classificação de
            novas notícias
```
