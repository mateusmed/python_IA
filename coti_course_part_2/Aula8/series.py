# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('AirPassengers.csv')
base.head()

print(base.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, 
                                               '%Y-%m')

base = pd.read_csv('AirPassengers.csv',
                   parse_dates = ['Month'],
                   index_col = 'Month', 
                   date_parser = dateparse)

base.head()


# Time series
ts  = base['#Passengers']       
ts


ts['1950-01-01' :  '1950-07-31']

ts.index.max()
ts.index.min()

plt.plot(ts)


# Agrupando os dados
# Codigo -> A, significa ano na função resample
ts_ano = ts.resample('A').sum()
plt.plot(ts_ano)


# Agrupando pelo mes
ts_mes = ts.groupby([lambda x:  x.month]).sum()
ts_mes
plt.plot(ts_mes)

# Decomposição
from statsmodels.tsa.seasonal import seasonal_decompose
decomposicao = seasonal_decompose(ts)

# Tendencia
decomposicao.trend
# Sazonalidade
decomposicao.seasonal
# Aleatorio
decomposicao.resid

# Plotar 
plt.plot(ts)
plt.plot(decomposicao.seasonal)
plt.plot(decomposicao.trend)
plt.plot(decomposicao.resid)

# Media
ts.mean()

# Media do ultimo ano
ts['1960-01-01' :  '1960-12-01'].mean()


media_movel = ts.rolling(window = 12).mean()
media_movel

plt.plot(ts)
plt.plot(media_movel, color='red')

# Previsões 
from statsmodels.tsa.arima_model import ARIMA

# order 3, parâmetros
# p -> numero de termos regresivos, q, numero de media movel
# d -> numero de diferencas nao sozonais

modelo = ARIMA(ts, order = (2, 1, 2))

# modelo de treinamento
modelo_treinado = modelo.fit()

modelo_treinado.summary()

previsoes = modelo_treinado.forecast(steps=12)[0]
previsoes

type(modelo_treinado)

eixo = ts.plot()

modelo_treinado.plot_predict('1960-01-01' , '1962-01-01',
                             ax = eixo, plot_insample = True)


# Instalar o pacote -> pyramid
# pip install pyramid
from pyramid.arima import auto_arima

modelo_auto = auto_arima(ts, m=12, seasonal = True, trace = True)

modelo_auto.summary()

# Prever os proximos 12 meses
proximos_12 = modelo_auto.predict(n_periods = 12)
plt.plot(proximos_12)

from datetime import datetime

test = pd.DataFrame(['1961-01-01', '1961-02-01', '1961-03-01', 
                     '1961-04-01', '1961-05-01', 
                               '1961-06-01', '1961-07-01', 
                               '1961-08-01' , '1961-09-01' ,'1961-10-01',
                              '1961-11-01', '1961-12-01'], columns = ['datas'] )

print(dateparse)
previsao  = pd.DataFrame(proximos_12,  columns=['Prediction'])
previsao.index = pd.to_datetime(test.datas)
previsao

pd.concat([ts, previsao], axis =1).plot()









































