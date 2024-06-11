# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:20:04 2019

@author: Aluno 09
"""

# esse script aqui é complicado tem q estudar

import pandas as pd
import numpy as np

df = pd.read_csv('primary-results.csv')

# candidatos
df['candidate'].unique()


# pivot - reorganização de dados tabulares
# transformando linhas em colunas,
# criando uma melhor organização dos dados
pd.pivot_table(df, 
               index = ['state', 'party', 'candidate'],
               values = ['votes'], 
               aggfunc = {'votes': np.sum})


df['rank'] = df.groupby(['county', 'party'])['votes'].rank(ascending=False)

df.head()

# agrupar os dados por estado partido, candidato, 
df_groupby = df.groupby(['state', 'party', 'candidate']).sum()

del df_groupby['fips']
del df_groupby['fraction_votes']

# reorganizando os indeces, exemplo reorganizando o 'vetor' de posições
df_groupby.reset_index(inplace = True)

df_groupby.head()

# refazendo o rank agrupando novamente por estado e partido
df_groupby['rank'] = df_groupby.groupby(['state', 'party'])['votes'].rank(ascending = False)


df_groupby.head()

pd.pivot_table(df_groupby, 
               index = ['state', 'party', 'candidate'],
               values = ['rank', 'votes'])



