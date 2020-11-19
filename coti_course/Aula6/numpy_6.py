# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:48:49 2019

@author: Aluno 09
"""
import numpy as np

# separador por default é o espaço
val1, val2, val3 = np.loadtxt('dados.txt', skiprows = 1, unpack = True)

val1
val2
val3

my_array = np.genfromtxt('dados2.txt', 
                         skip_header = 1,
                         filling_values = 1000) 
# para todos os valores faltantes, preencha com 1000

my_array


valores = np.genfromtxt('arquivo.csv', 
                        delimiter = ';', 
                        skip_header=1)


valores

data = np.genfromtxt('iris.data', 
                     delimiter = ',', 
                     usecols = (0, 1, 2, 3))
