import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report,confusion_matrix

#base de dados ficticia
dados = pd.read_csv(r'C:\Users\andre\OneDrive\Área de Trabalho\Python\IA\cap13_Class\advertising.csv')

#------------------Convertendo dados categoricos em numeros------
Ad_topic_line = pd.get_dummies(dados['Ad Topic Line'],drop_first=True)
dados.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)

#------------------Criando conjunto de dados para treinar a IA---
X = dados.drop('Clicked on Ad',axis=1)
Y = dados['Clicked on Ad']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=101)

#----------------Regressao logica--------------------------------
logmodel = LogisticRegression(solver = 'liblinear')
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)

precisao = classification_report(Y_test,predictions)
c_mat = pd.DataFrame(confusion_matrix(Y_test,predictions))

#estudar a influencia de cada variavel
coef = pd.DataFrame((np.exp(logmodel.coef_.T)-1)*100, index=X.columns, columns=['Odds Ratio'])

sns.heatmap(c_mat,annot=True)
plt.show()
sns.heatmap(coef,annot=True)
plt.show()

#conclusao
print(f'Variável mais relevante: {coef['Odds Ratio'].idxmax()}\nA cada 1 unidade de aumento em {coef['Odds Ratio'].idxmax()} a chance da pessoa clicar no anúncio eleva-se em {coef['Odds Ratio'].max():.1f} %')
