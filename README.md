# Kaggle - Porto Seguro Safe Driver Prediction

## Descrição

Este projeto tem como objetivo prever a probabilidade de um cliente registrar um sinistro de seguro automotivo no próximo ano, utilizando técnicas de machine learning.

O problema faz parte de uma competição do Kaggle proposta pela Porto Seguro, onde previsões mais precisas permitem ajustar melhor os preços dos seguros, tornando-os mais justos para os clientes.

A métrica de avaliação utilizada é o **Gini Normalizado**, diretamente relacionada à AUC:

Gini = 2 * AUC - 1

---

## Abordagem

A solução utiliza **ensemble learning**, combinando diferentes modelos para capturar padrões complementares nos dados.

### Modelos Base

* **XGBoost**
* **LightGBM**
* **Gaussian Naive Bayes**

Cada modelo contribui de forma diferente:

* XGBoost: robusto e conservador
* LightGBM: mais agressivo, captura padrões complexos
* GaussianNB: modelo probabilístico, aumenta diversidade

---

## Stacking

O projeto utiliza **stacking (empilhamento)** para combinar os modelos.

### Pipeline

1. Treinamento dos modelos base com validação cruzada estratificada
2. Geração de previsões **Out-of-Fold (OOF)**
3. Uso das previsões como novas features
4. Treinamento de um **meta-modelo (Logistic Regression)**
5. Geração da predição final

### Por que funciona?

O stacking permite que um modelo aprenda:

* quando confiar mais em cada modelo base
* como corrigir erros individuais
* como combinar diferentes padrões aprendidos

---

## Otimização

* **Optuna** para busca de hiperparâmetros
* **Seleção de features** (quando aplicável)
* Validação cruzada estratificada em todas as etapas

---

## Resultados

### Modelos Base (OOF)

| Modelo     | Gini    |
| ---------- | ------- |
| XGBoost    | 0.28629 |
| LightGBM   | 0.28392 |
| GaussianNB | 0.23534 |

### Stacking

| Métrica            | Valor   |
| ------------------ | ------- |
| Gini (OOF)         | 0.28683 |
| Kaggle (Public LB) | 0.28126 |

---

## Benchmark da Competição

Melhor solução no Kaggle:

* **Gini: 0.29698**

O resultado obtido neste projeto é competitivo e próximo ao topo, considerando uma abordagem relativamente simples e bem estruturada.

---

## Considerações Finais

Este projeto evidencia a força de técnicas de ensemble em problemas tabulares.

Ao invés de focar apenas em um modelo, a combinação estratégica de múltiplos algoritmos permite:

* melhor generalização
* maior robustez
* ganhos consistentes de performance

Em problemas reais, essa abordagem costuma ser mais eficaz do que ajustes isolados em modelos individuais.
