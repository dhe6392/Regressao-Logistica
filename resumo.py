import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report,confusion_matrix

test = pd.read_csv(r'C:\Users\andre\OneDrive\Área de Trabalho\Python\IA\cap13_Class\titanic_test.csv')
train = pd.read_csv(r'C:\Users\andre\OneDrive\Área de Trabalho\Python\IA\cap13_Class\titanic_train.csv')

'''colunas_numericas = []
for coluna in train.keys():
    if train[coluna].dtypes in ['float64','int64']:
        colunas_numericas.append(coluna)

sns.heatmap(data=train.corr(numeric_only=True),annot=True)
sns.countplot(data=train,x='Survived',hue='Sex')
sns.countplot(data=train,x='Survived',hue='Pclass')
sns.displot(data=train['Age'].dropna(),kde=True,bins=30)
sns.catplot(data=train['Fare'],kind='count')'''

#----------------------------Limpeza----------------------------------
idade_media = train.groupby('Pclass').mean(numeric_only=True)['Age']

for index,row in train.iterrows():
    if pd.isna(row['Age']):
            train.at[index,'Age'] = idade_media[row['Pclass']]

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'],drop_first=True)#cria uma coluna binaria para cada valor de sex e depois remove a primeira, pra evitar redundancia
embark = pd.get_dummies(train['Embarked'],drop_first=True)

colunas = 'Pclass Age SibSp Parch'.split()
X = pd.concat([train[colunas],sex,embark],axis=1)
Y = train['Survived']
dados = pd.concat([X,Y],axis=1)

#aqui começa a treinar a IA
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

#----------------------------Linear----------------------------------
lm = LinearRegression()
lm.fit(X_train,Y_train)
itc = lm.intercept_
coef = pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coefficient'])
predictions = lm.predict(X_test)

#print(itc)
#print(coef)

#checando se ha multicolinearidade entre as variaveis internas. Se VIF<5 ta de boa
X = X.astype(float)
vif = pd.DataFrame()
vif['variavel'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
#print(vif)


#----------------------------Lógica----------------------------------
logmodel = LogisticRegression(solver = 'liblinear')
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)

precisao = classification_report(Y_test,predictions)
c_mat = pd.DataFrame(confusion_matrix(Y_test,predictions))#matriz de VP,VN,Fp e Fn
print(precisao)


plt.show()

'''
Regressao linear: indica quanto a variavel Y muda quando X se altera em 1 unidade.
Difere de pearson pq pearson considera apenas o par de pontos (X,Y), q é igual a (Y,X)
enquanto a regressao considera a influencia de uma variavel quando as outras estao constantes,
e leva todas elas em conta na regressao.
Regressao logistica: calcula probabilidades, nao valores absolutos como na linear. Mais sensato
para quando o Y é binario e vc quer saber a chance de ser 1 (ou 0) a partir das variaveis
Acuracy: acertos totais (VP+VN)/total
Precision: VP/(VP+FP). Quantos dos previstos vivos realmente estavam vivos 
Recall: VP/(VP+FN). Quanto do total de vivos foram previstos como vivos
F1-score: media harmonica entre precision e recall
support: total
'''