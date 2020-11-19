# -*- coding: utf-8 -*-

import pandas as pd
from db import DemoDB

database =  DemoDB()
database.tables

# Carregando as tabelas album e artista
album = database.tables.Album.all()
artist = database.tables.Artist.all()

album.head()

artist.head()

# Merge
album_artist = pd.merge(artist, album)

album_artist.head()

# Campos iguais
album_artist = pd.merge(artist, album, on="ArtistId")

# Campos diferentes
album.rename(columns = {'ArtistId' :  'Artist_Id'}, inplace = True)

album.head()

album_artist = pd.merge(artist, album, left_on='ArtistId', 
                        right_on= 'Artist_Id')


album_artist = pd.merge(artist, album, left_on='ArtistId', 
                        right_on= 'Artist_Id').drop('Artist_Id', 
                                             axis =  1)


album_artist.head()


# Conjunto de vendas
# Unir os dados dos setores
vendas1 = pd.DataFrame({
        'nome' : ['Lucas', 'Vinicius'],
        'codigo' :  [10, 20]
        })

vendas2 = pd.DataFrame({
        'nome' :  ['Ana', 'Vinicius', 'Joana'],
        'valor' : [5000, 3500, 2020]
        })


vendas_total = pd.merge(vendas1, vendas2, 
                        on="nome", how="inner")

vendas_total

vendas_total = pd.merge(vendas1, vendas2, 
                        on="nome", how="outer")

vendas_total

vendas_total = pd.merge(vendas1, vendas2, 
                        on="nome", how="left")

vendas_total


vendas_total = pd.merge(vendas1, vendas2, 
                        on="nome", how="right")

vendas_total


























