# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregadndo os dados que serão usados
base = pd.read_csv('cars.csv')
base.head()

base = base.drop('Unnamed: 0', axis = 1)
base.head()

# Dist -> vetores
x = base.iloc[:, 1].values

# Speed -> vetores
y  = base.iloc[:, 0].values


# Correlação
print(np.corrcoef(x, y))


# -> matriz -> converter os dados para matriz
x  = x.reshape(-1, 1)
x

# Modelo de regressão linear 
modelo = LinearRegression()
modelo.fit(x, y)

# Informações do modelo criado
modelo.intercept_
modelo.coef_

# Predição -> dist -> 22
modelo.intercept_ + modelo.coef_ * 22


# previsão
modelo.predict([[22], ])














