# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:45:16 2019

@author: Aluno 09
"""


# numero de respostas == numero de neuronios
# ativação da saida 1
# desligamento de saida 0
# numero de loops é igual a época;

from sklearn.neural_network import MLPClassifier
from sklearn import datasets
iris = datasets.load_iris()

entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose = True,
                           max_iter = 2000, # numero de iterações
                           tol = 0.00001,   # tolerancia a erro
                           learning_rate_init = 0.01) # taxa de aprendizado inicial

redeNeural.fit(entradas, saidas)

redeNeural.predict([entradas[120]])

 



