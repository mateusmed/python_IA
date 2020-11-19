# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:36:46 2019

@author: Aluno 09
"""

import numpy as np

l = [10, 20, 30, 40]

print(l)

# Converte a lista
a = np.array(l)
a

# chamdo de fatia total - é um ponteiro para o array a
b = a[:]
b

b[0] = 1000
# a também sofreu alteração
a

c = a.copy()
c
c[0] = 10
c

a