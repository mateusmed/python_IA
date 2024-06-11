# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:05:24 2019

@author: Aluno 09
"""

import pandas as pd
from apyori import apriori


dados = pd.read_csv('mercado2.csv', header = None)
dados

len(dados)

transacoes = []

# lista de tranções
# [[café, leite, pão], [café, leite, ovos]]

for i in range(len(dados)):
    transacoes.append([str(dados.values[i, j]) for j in range(len(dados.columns))])

transacoes

# regras
# suporte e confiança <- relembrar como isso funciona

regras =  apriori(transacoes, 
                  min_support = 0.003, #
                  min_confidence = 0.2,  # % de ocorrencias em quantas transações está
                  min_lift = 3, # quem compra x tem y% de chande de levar
                  min_lenght = 2) # 

resultados = list(regras)

# -- 154
len(resultados)

#RelationRecord(items=frozenset({'light cream', 'chicken'}), 
#support=0.004532728969470737, 
#ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), 
#                                     items_add=frozenset({'chicken'}), 
#                                     confidence=0.29059829059829057, 
#                                     lift=4.84395061728395)])

resultados[0]

resultados2 = [list(x) for x in resultados]
resultados2[0]

resultadoFormatado = []

for j in range(0, 20):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])

for r in resultadoFormatado:
    print("================")
    print(r)
    print("================")





