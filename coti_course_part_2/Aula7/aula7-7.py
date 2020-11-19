# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:44:29 2019

@author: Aluno 09
"""
import pandas as pd
copacabana = pd.read_csv('copacabana_lat_lng.csv', delimiter = ',')

copacabana

# indexação boleana
# retorna um vetor com as condições logicas de todo o conjunto de dados
copacabana['Quartos'] > 5

condicao = copacabana['Quartos'] > 5

copacabana[condicao]

copacabana['Valor Total'] = copacabana['AreaConstruida'] * copacabana['VAL_UNIT']

copacabana['Valor Total']