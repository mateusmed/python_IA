# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:20:47 2019

@author: Aluno 09
"""
import numpy as np

a = np.array([[1, 2], [3, 4]])
a

# repete cada valor em 3 vezes - não mantem a estrutura dos dados
# retornou um array e não uma lista
np.repeat(a, 3)

# axis = 0 LINHA
np.repeat(a, 2, axis = 0)

# axis = 1 COLUNA
np.repeat(a, 2, axis = 1)

# Tile - repetir

a = np.array([1, 2, 3])
a

# repete a sequencia;
np.tile(a, 2)

# dividindo array -> split
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

b

# split em 4 linhas.
# separou as linhas,
arrays = np.split(b, 4, axis = 0)

for array in arrays:
    print('=======================')
    print(array)

#arrays de zeros
    
np.zeros(4)

# tupla 2 linhas 3 colunas
np.zeros((2, 3))
# == ?
np.zeros([2, 3])

# mnatriz identidade
np.eye(3)

# indexação boleana
a = np.array([[1, 2], [3, 4], [5, 6]])

a

a > 3

a [a > 3]

# numeros complexos com numpy j representa um numero complexo
a = np.array([1, 10+2j, 20 + 5j], dtype = complex)

a

a[1] + a[0]

a[1] - a[0]

#geração de arrays
np.arange(10)
np.arange(1, 100, 2)

# lim spaces
np.linspace(2.0, 3.0, 5)


# distinct - elementos unicos
a = np.array([[1, 2], [2, 3], [3, 3], [4, 4]])

a

# me de os elementos unicos
np.unique(a)

# embaralhamento.
np.random.shuffle(a)
a










