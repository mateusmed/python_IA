# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:48:38 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
# transformar em dados categóricos apenas os dados que não são a classificação
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

x_treinamento

len(x_treinamento)
len(x_teste)

len(y_treinamento)
len(y_teste)

naive = GaussianNB()

# treinamento
naive.fit(x_treinamento, y_treinamento)

#validação
previsoes = naive.predict(x_teste)
previsoes

from sklearn.metrics import confusion_matrix, accuracy_score

matriz = confusion_matrix(previsoes, y_teste)
matriz

# precisão
# precisão de 71 %
accuracy_score(y_teste, previsoes)

# fazendo a previsão -  classificando entre good e bad.
previsoes = naive.predict(x_teste)
previsoes





