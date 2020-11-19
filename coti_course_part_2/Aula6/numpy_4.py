# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:44:18 2019

@author: Aluno 09
"""

import numpy as np

# Inserindo dados no array
arr = np.array([1, 2, 3])

arr

# inserindo 10 na posição 1
np.insert(arr, 1, 10)

a = np.array([[1, 2, 3], [4, 5, 6]])

a

#somatorio de matriz
a.sum()

# inserindo o 5 na posição 1, mas como é uma 
#matriz ele insere pela linha ou coluna
# axis = 0 -> LINHA
np.insert(a, 1, 5, axis = 0)

# inserindo o 5 na posição 1, mas como é uma 
#matriz ele insere pela linha ou coluna
# axis = 1 -> COLUNA
np.insert(a, 1, 5, axis = 1)

# Juntando arrays
a1 = np.array([1, 2, 3])

np.append(a1, [4, 5, 6])

a = np.array([[1, 2] , [3, 4]])
a

# adicionando linhas
np.append(a, [[5, 6]], axis = 0)

# adicionando colunas
np.append(a, [[5], [6]], axis = 1)


#transposta
a.transpose()

a.T




