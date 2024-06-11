# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:32:01 2019

@author: Aluno 09
"""
import pandas as pd
import numpy as np

# pandas
#Series - dados em uma unica dimensão
s = pd.Series([1, 4, 5, 6, 7, 10, 6])

s

s[0]

s[2:4]

# Describe -> summary do R
s.describe()

# media
s.mean()

# mediana 
s.median()

# elemento duplicados
s.duplicated()

s2 = pd.Series([11, 5, 8])

s2

# Append - juntar. 
s.append(s2)

# atenção ao apendar, pois o indice não se altera, dado isso 
s

# Dataframe -> criação
df = pd.DataFrame([
                   ['Python web', 2000],
                   ['Machine learning', 3000],
                   ['Lógica de programação', 4500]
                  ])

df.shape
df

df = pd.DataFrame([
                   ['Python web', 2000],
                   ['Machine learning', 3000],
                   ['Lógica de programação', 4500]                   
                  ], columns = ['curso', 'alunos'])



df

df['curso']
df['alunos']

df['alunos'].mean()
df['alunos'].median()

















