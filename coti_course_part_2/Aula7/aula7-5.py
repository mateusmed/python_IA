# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:11:49 2019

@author: Aluno 09
"""

# pip install db.py
from db import DB

# carregando os dados do sqlite para o python
database = DB(filename = 'logs.sqlite3', dbtype = 'sqlite')

database

# exibir as tabelas
database.tables

# carregando tabela # pegando a estrutura dos dados.
log_df = database.tables.log

# vazio
log_df

# consulta 
log_df = database.tables.log.all()

log_df

# query - consulta
log_df = database.query('select * from log where user_id = 3')
log_df



