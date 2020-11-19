# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:57:26 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np

df = pd.DataFrame([['PE', 'Pernambuco', 'recife'],
                  ['RJ', 'Rio de Janeiro', 'rio de janeiro'],
                  ['SP', 'São paulo', 'são paulo'], 
                  ['PD', 'Paraíba', 'joão pessoa'],
                  ['MG', 'Minas Gerais', 'belo horizonte'],
                  ['CE', 'Ceara', 'fortaleza']],
                  columns = ['sigla', 'nome', 'capital'])

df

df['sigla']

# Index
df.index

# pegar pelo index
df.ix[0]

# posição
df.iloc[0]
# pega o intervalo entre 0 e 2 log, 0 e 1 , não pega a ultima posição
df.iloc[0:2]

df.loc[0]
# pega a ultima posição
df.loc[0:2]


# substituir o indice do dataframe
df.index = df['sigla']
df

# removendo a coluna, pois colocamos a coluna sigla como index
# portanto ela não deveria estar duplicada;
del df['sigla']
df

# buscando pelo novo indice criado.
df.loc['PE':'MG']











