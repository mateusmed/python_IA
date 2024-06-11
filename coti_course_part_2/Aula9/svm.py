# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:48:47 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#svm - support vector machine
from sklearn.svm import SVC

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

# SVC -> suport vector machine

# Treinamento
classificador = SVC(kernel = 'linear', 
                    random_state = 1)

classificador.fit(x_treinamento, y_treinamento)


# teste -> previsoes
previsoes = classificador.predict(x_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

matriz = confusion_matrix(y_teste, previsoes)
prec1 = accuracy_score(y_teste, previsoes)

prec1

# seleção de atributos
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier()
forest.fit(x_treinamento, y_treinamento)
importancias = forest.feature_importances_
importancias

# construir um dataframe com as importancias
# pegar os campos que são mais relevantes para o calculo.

campos = np.array(credito.columns[0:20].values)

df = pd.DataFrame(importancias, campos)
df

# ordenando por importancia

df = pd.DataFrame(importancias, campos)

df = pd.DataFrame({ 'importancia': importancias, 
                    'campo': campos})

# inplace mantem a alteração na mesma variavel, no caso -> df
df.sort_values(by = ['importancia'], inplace = True, ascending = False)

df


# pegando as 4 primeiras linhas e criando um novo treinamento.
x_treinamento2 = x_treinamento[:, [0, 1, 2, 3]]

x_teste2 = x_teste[:, [0, 1, 2, 3]]

classificador2 = SVC(kernel = 'linear', 
                     random_state = 1)

# mantendo y - quantidade de registros maior que o x_treinamento2
classificador2.fit(x_treinamento2, y_treinamento)

previsoes = classificador2.predict(x_teste2)

matriz = confusion_matrix(y_teste, previsoes)
matriz

prec2 = accuracy_score(y_teste, previsoes)
prec2

# classificação equivocada, 
# foi classificado todos com good, isso significa que os 4 primeiros campos
# usados não são relevantes para o treinamento.









