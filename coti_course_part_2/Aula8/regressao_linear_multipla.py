# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

base = pd.read_csv('mt_cars.csv')
base = base.drop('Unnamed: 0', axis = 1)

base.head()


# X -> cyl, disp, hp
x = base.iloc[:, 1:4].values
x

# y -> mpg
y = base.iloc[:, 0].values
y

# Regressão multipla
modelo = LinearRegression()
modelo.fit(x, y)

modelo.score(x, y)

# predição
registro = np.array([[4, 200, 100]])
registro

# MPG -> resultado
modelo.predict(registro)














