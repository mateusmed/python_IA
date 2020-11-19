# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:37:27 2019

@author: Aluno 09
"""

import pandas as pd
import pydataset

# Dados categoricos
titanic = pydataset.data('titanic')

# primeiras linhas
titanic.head(10)

# colunas
titanic.columns

# verificando quantidade de bytes
titanic['class'].nbytes

type(titanic['class'])

# transformar em dados categoricos do pandas
# reduzindo o tamanho do arquivo dado que criamos um indice
# uma nova referencia para a coluna class
# "para não repetir a string nas colunas"
titanic['class'] = titanic['class'].astype('category')

# confirmando a redução de tamanho do arquivo.
titanic['class'].nbytes