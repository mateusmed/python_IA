# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:54:21 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

# tipo de treinamento não supervisionado com intuito de fazer agrupamento

from sklearn import datasets


iris = datasets.load_iris()

#atributos
iris.data

#classe
iris.target

unicos, quantidade = np.unique(iris.target, 
                               return_counts = True)

unicos 
quantidade

# numero de cluster === numero de classes
cluster = KMeans(n_clusters = 3)

cluster.fit(iris.data)

# kmeans cria a media;
# centroides
centroides = cluster.cluster_centers_
centroides

# previsoes
previsoes = cluster.labels_
previsoes

# exibir a matriz de confusao
resultados = confusion_matrix(iris.target,
                              previsoes)

# os resultados são agrupamentos
# portanto não temos um nome bem definido
resultados


plt.scatter(iris.data[previsoes == 0, 0],
            iris.data[previsoes == 0, 1],
            c = 'green', label = 'classe A')

plt.scatter(iris.data[previsoes == 1, 0],
            iris.data[previsoes == 1, 1],
            c = 'red', label = 'classe B')

plt.scatter(iris.data[previsoes == 2, 0],
            iris.data[previsoes == 2, 1],
            c = 'blue', label = 'classe C')

plt.legend()












