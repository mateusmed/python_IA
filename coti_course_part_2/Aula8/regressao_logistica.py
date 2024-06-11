# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

base = pd.read_csv('Eleicao.csv', sep = ';')
base.head()

# Grafico
plt.scatter(base.DESPESAS, base.SITUACAO)

# Estatistica básica
base.describe()

print(np.corrcoef(base.DESPESAS, base.SITUACAO))

# Base -> DESPESAS
x  = base.iloc[:, 2].values

# Y -> Situacao
y  = base.iloc[:, 1].values

x
y

# Transformar em matriz
x = x[:, np.newaxis]
x
x.shape


#Criação do modelo de regressão logistica
modelo = LogisticRegression()
modelo.fit(x, y)

# Parâmetros do modelo
print(modelo.intercept_)
print(modelo.coef_)

# Calculo do sigmod de x
def model(x):
    return 1 / (1 + np.exp(-x))

x_teste = np.linspace(10, 3000)
x_teste


# Ravel -> numpy array matriz em vetor
r = model(x_teste * modelo.coef_ + modelo.intercept_).ravel()
r
plt.scatter(x, y)
plt.plot(x_teste, r, c='red')

# Utilizar o arquivo novos candidatos
base_previsores = pd.read_csv('NovosCandidatos.csv', sep=';')
base_previsores
despesas = base_previsores.iloc[:, 1].values
despesas
despesas = despesas[:, np.newaxis]
despesas

previsoes_teste = modelo.predict(despesas)
previsoes_teste

base_previsores['SITUACAO'] = previsoes_teste
base_previsores






















































