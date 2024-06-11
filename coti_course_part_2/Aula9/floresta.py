# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:48:56 2019

@author: Aluno 09
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:38:40 2019

@author: Aluno 09
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:48:20 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#floresta
from sklearn.ensemble import RandomForestClassifier

credito = pd.read_csv('Credit.csv')
credito.head()

len(credito)

credito.shape


# Classe ->
# pegando a posição da coluna 20
classe = credito.iloc[:, 20].values

classe

# previsoes ->
# pegando de 0 à 19
previsores = credito.iloc[:, 0:20].values
previsores[:2]

#efetuar o pré-processamento
# possibilidades de dados na coluna 'checking_status'
credito['checking_status'].unique()

# TRANSFORMAR DE CATEGORICO PARA DISCRETO.
# TRANSFORMAR O TEXTO EM UM DADO NUMERO PARA O PROCESSAMENTO
# o algoritmo só funciona com dados numericos
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 11])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 13])
previsores[:, 14] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 16] = labelencoder.fit_transform(previsores[:, 16])
previsores[:, 18] = labelencoder.fit_transform(previsores[:, 18])
previsores[:, 19] = labelencoder.fit_transform(previsores[:, 19])

#verificando a substituição
previsores

# sample de train, test
# previsores X
# class Y
# random_state é a semente randomica
# 70 treino
# 30 teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores, 
                                                                  classe, 
                                                                  test_size = 0.3, 
                                                                  random_state = 0)

# distancia manhathan 
# distancia euclidiana

len(x_treinamento)
len(x_treinamento)



# criação do RandomFlorest
# n_estimators é o número de arvores
floresta = RandomForestClassifier(n_estimators = 100, 
                                  criterion = "entropy")

floresta

floresta.fit(x_treinamento, y_treinamento)

from sklearn.metrics import confusion_matrix, accuracy_score

# validação -> teste
previsoes = floresta.predict(x_teste)


matriz = confusion_matrix(y_teste, previsoes)
matriz

accuracy_score(y_teste, previsoes)



















