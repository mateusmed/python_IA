# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:26:27 2019

@author: Aluno 09
"""

import numpy as np

# calculos de tempo de processamento.

int_range = 100000

array = np.arange(int_range )
array

lista = list(range(int_range))

# arrau numpy muito mais rapido
%time for _ in range(10): array = array * 2

%time for _ in range(10): lista = [x * 2 for x in lista]

# Array -> multidimensional (gerado de forma randomica)
data = np.random.rand(2, 3)
data

#formato do array
data.shape