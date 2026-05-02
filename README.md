# Kaggle - Porto Seguro Safe Driver Prediction

## Descrição

Este projeto tem como objetivo prever a probabilidade de um cliente registrar um sinistro de seguro automotivo, utilizando técnicas de machine learning. A métrica de avaliação utilizada é o coeficiente de Gini.

---

## Estrutura do Projeto

- `data_sets/`: arquivos de treino e teste (não versionados)
- `utils/`: arquivos JSON com parâmetros dos modelos
- `submissions/`: arquivos de submissão gerados
- `main.py`: script principal com pipeline de treino e stacking

---

## Modelos Utilizados

Foram utilizados três modelos base:

- XGBoost
- LightGBM
- Gaussian Naive Bayes

Cada modelo pode utilizar:
- conjunto completo de features
- subconjunto selecionado de features (definido nos arquivos JSON)

---

## Stacking

O projeto utiliza uma abordagem de stacking, que consiste em combinar múltiplos modelos para melhorar o desempenho final.

### Como funciona

1. Os modelos base (XGB, LGB, NB) são treinados utilizando validação cruzada.
2. São geradas previsões out-of-fold (OOF) para cada modelo.
3. Essas previsões são utilizadas como novas features.
4. Um modelo meta (Logistic Regression) é treinado sobre essas novas features.
5. O modelo final combina os outputs dos modelos base.

### Vantagens

- Redução de overfitting
- Combinação de diferentes padrões aprendidos pelos modelos
- Melhora da performance final

---

## Otimização

- Os parâmetros dos modelos base armazenados nos arquivos JSON foram previamente obtidos por meio da otimização individual de cada modelo.
- O modelo meta (Logistic Regression) é otimizado com Optuna.
- A otimização é feita com validação cruzada estratificada.

---

## Resultados

### Gini dos Modelos Base (OOF)

| Modelo        | Gini Normalizado |
|--------------|------------------|
| XGBoost      | 0.28629          |
| LightGBM     | 0.28392          |
| GaussianNB   | 0.23534          |

### Stacking

| Métrica              | Valor    |
|---------------------|---------|
| Gini (OOF)          | 0.28683 |
| Kaggle (Public LB)  | 0.28126 |