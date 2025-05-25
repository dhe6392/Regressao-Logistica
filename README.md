# Regressão Logística: Previsão de Cliques em Anúncios

Este projeto utiliza **regressão logística** para prever se um usuário irá clicar ou não em um anúncio online com base em características como idade, tempo gasto no site, renda, sexo e tempo na internet. Encontrou-se que a cada 1 ano de aumento na idade eleva-se em 23.7% a chance de uma pessoa clicar no anúncio, com baixa relevância absoluta para as outras variáveis.

## 📁 Arquivos

- `projeto.py`: script principal com todo o processo de:
  - Carregamento e limpeza dos dados
  - Separação em treino e teste
  - Treinamento do modelo de regressão logística
  - Avaliação com matriz de confusão e métricas (precision, recall, F1-score)
  - Interpretação de **odds ratio** dos coeficientes
  - Visualização com heatmaps

- `advertising.csv`: conjunto de dados com registros simulados de usuários e suas interações com anúncios.

## 📊 Bibliotecas utilizadas

- `pandas`
- `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

## 📈 Resultado

O modelo fornece uma matriz de confusão e um relatório de classificação que mostram a performance na previsão de cliques.  
Também exibe os **fatores mais influentes** para a tomada de decisão usando o **odds ratio** da regressão logística.

