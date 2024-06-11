# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:53:39 2019

@author: Aluno 09
"""

import pandas as pd
import numpy as np

df = pd.read_csv('primary-results.csv')

df.head()

# total de zonas
len(df)

# quantidade de votos por candidatos
df['votes'].sum()

# total de votos Hillary
condicao = (df['candidate'] == 'Hillary Clinton')
df[condicao]['votes'].sum()

# total de votos pro trump
condicao_2 = (df['candidate'] == 'Donald Trump')
df[condicao_2]['votes'].sum()

# agrupando os dados
df.groupby('candidate').aggregate({'votes': [min, np.mean, max]})

# total de votos por estado
condicao_3 = df['state_abbreviation'] == 'AL'

df [condicao_3]['votes'].sum()

# agrupando por mais de um campo.
df.groupby(['state_abbreviation', 'candidate'])['votes'].sum()






