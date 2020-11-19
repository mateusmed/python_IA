# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:16:44 2019

@author: Aluno 09
"""

import pandas as pd

# instalar -> pip install pydataset - rodar o prompt da anaconda
# pacote com datasets para estudo
import pydataset

pydataset.data()

len(pydataset.data())


titanic = pydataset.data('titanic')

titanic

# head - 5 primeiros elementos
titanic.head()

# tail - 10 ultimos elementos
titanic.tail(10)

# contagem de dados
titanic['class'].value_counts()

titanic['age'].value_counts()




