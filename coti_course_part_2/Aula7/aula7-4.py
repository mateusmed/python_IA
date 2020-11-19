# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:38:41 2019

@author: Aluno 09
"""

import pandas as pd

# carregando um arquivo XLS
populacao = pd.read_excel('total-populacao-pernambuco.xls')

populacao

#total de mulheres
populacao['Total de mulheres']

# proporção de mulheres e homens
populacao['Total de mulheres'] / populacao['Total de homens']

# menor proporção
min_proporcao = (populacao['Total de mulheres'] / populacao['Total de homens']).min()
min_proporcao

# maior proporção
(populacao['Total de mulheres'] / populacao['Total de homens']).max()

# criando compo de proporção
populacao['Proporcao'] = populacao['Total de mulheres'] / populacao['Total de homens']

# achando o registro com o menor proporcao entre homens e mulheres
municipio = populacao[ populacao['Proporcao'] == min_proporcao]
municipio['Nome do município']


