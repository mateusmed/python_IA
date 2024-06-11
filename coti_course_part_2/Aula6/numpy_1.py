# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:35:35 2019

@author: Aluno 09
"""
# instalar anaconda que já vem com:  (python, spyder, jupter)
# vem com o pacote de machine learn
# (numpy -> para arrays, conceitos de algebra linears)
# (pandas -> para data frames)
# (matplotlib -> para gráficos)

import numpy

a = numpy.array([10, 20, 30, 40])

print(a)

# tipo de dados

type(a)

#matriz
mat = numpy.array([[1, 2], [3, 4]])
mat

# elemento nas posições
# primeira posição / linha 0 e coluna 0
mat[0, 0]
# ultima posição / linha 1 e coluna 1
mat[1, 1]

#indexação negativa 
print(mat[-1, -1])

# duas matrizes
m1 = numpy.array

# duas matrizes
m1 = numpy.array([[1, 2], [3, 4]])
m2 = numpy.array([[5, 6], [7, 8]])

print(m1 + m2)
print(m1 - m2)
print(m1 * m2)
print(m1 / m2)

m3 = numpy.array([1, 2, 3, 4])

m3.sum()

# Indice da posição do maior elemento
print(m3.argmax())

# pegar o valor na posição
p = m3.argmax()
m3[p]










