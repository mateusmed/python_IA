# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:21:42 2019

@author: Aluno 09
"""

# pip install scikit-fuzzy

from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import skfuzzy


iris = datasets.load_iris()

# Aplicar a transposta na matriz de dados

r = skfuzzy.cmeans(data = iris.data.T, 
                   c = 3,  # numero de grupos
                   m = 2,  # numero minimo de grupos
                   error = 0.005, 
                   maxiter = 1000, 
                   init = None)

# r -> retorna uma tupla com 7 posições
# a posição de indice 1, conjunto da dados de probabilidade
previsoes_porcentagem = r[1]

# c means cria um vetor de probabilidade

# primeiro elemento do conjunto
# [classe][item] - por causa da transposta
# avalie em qual dessas classes é a linha 0 

print(previsoes_porcentagem[0][0]) # 0.9966236296144394
print(previsoes_porcentagem[1][0]) # 0.0010719348596127498
print(previsoes_porcentagem[2][0]) # 0.0023044355259479164

#encrontrar os grupos que os elementos pertencem
previsoes = previsoes_porcentagem.argmax(axis = 0)
previsoes




