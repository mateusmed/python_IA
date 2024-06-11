# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:37:09 2019

@author: Aluno 09
"""

#pip install pyclustering

#medoids conculo de distancia


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

from sklearn import datasets
iris = datasets.load_iris()

# iris data -> somente duas colunas
# indice -> das medoids [3, 12, 20]

# usando apenas as duas primeiras colunas; 0 e 1
cluster = kmedoids(iris.data[:, 0:2],
                   [3, 12, 20]) # definido aleatoriamente um grupod e medoids

#efeturar o treinamento
cluster.process()

#previsoes
previsoes = cluster.get_clusters()
previsoes

# previsoes - 3 listas
# 
len(previsoes[0])
len(previsoes[1])
len(previsoes[2])

# medoids
medoids = cluster.get_medoids()
medoids


# Efetuar visualização dos dados

v = cluster_visualizer()
v.append_clusters(previsoes, iris.data[:, 0:2])

v.append_cluster(medoids, iris.data[:, 0:2], 
                 marker = '*', 
                 markersize = 15)


v.show()











