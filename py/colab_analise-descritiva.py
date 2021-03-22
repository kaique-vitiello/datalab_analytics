# -*- coding: utf-8 -*-

# @title Importar as bibliotecas e carregar os dados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import re
#from bokeh.charts import Histogram, show
import numpy as np

dataset = pd.read_csv('/content/AULA_BASE_FULL.txt', sep='\t') # Separador TAB
dataset.head(5)

#@title Avaliar a quantidade de dados preenchidos em cada variavel
dataset.info()

#@title Avaliar as variáveis quantitativas (Avg, Desvio Padrão, Max e Min)
dataset.describe().round()

#@title Analise de outliers na variavel VLR_PGTO x Sexo
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(8,6))

graph = sns.boxplot(x='Sexo', y='VLR_PGTO', data=dataset, orient="v")

#@title Eliminamos dados nulos e validamos que o maior volume de contratos está concentrado no valor de até R$2000,00.

filter_data = dataset.dropna(subset=['VLR_PGTO'])
plt.figure(figsize=(10,8))
ax = sns.histplot(filter_data['VLR_PGTO'], color='g', kde=False)

#@title Analisamos a frequência relativa por sexo dos devedores

filter_data = dataset.dropna(subset=['Sexo'])
type_counts = filter_data['Sexo'].value_counts()

dataset2 = pd.DataFrame({'Sexo': type_counts}, index = ['M', 'F'])
dataset2.plot.pie(y='Sexo', figsize=(8,8), autopct='%1.1f%%')

#@title Analisamos a quantidade de devedores por estado Civil, podemos concluir que Casado x Solteiro estão com valores bem proximos.

plt.figure(figsize=(10,5))
graph = sns.countplot(x='EstadoCivil', data=dataset)

plt.figure(figsize=(10,5))
graph = sns.countplot(x='Carteira', data=dataset)

# @title Converte a data de nascimando para "datetime" depois, calcula a idade com base na data de nascimento e o dia de hoje, em seguida cria uma nova coluna com a idade de cada pessoa.

dataset['DataNascimento'] = pd.to_datetime(dataset['DataNascimento'])
dataset['Idade'] = (datetime.now() - dataset['DataNascimento']) / 365
dataset['Idade'] = (dataset['Idade']).dt.days
dataset['Idade'].value_counts()

#@title Mostra em histograma a quantidade geral por idade
filter_data = dataset.dropna(subset=['Idade'])
plt.figure(figsize=(10,8))
ax = sns.histplot(filter_data['Idade'], color='g', kde=True)

#@title Analise de outliers nas variaveis Carteira x Idade
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(8,6))

graph = sns.boxplot(x='Carteira', y='Idade', data=dataset, orient="v")

#@title Analise de outliers nas variaveis QTDPGTO x Idade
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(10,8))
graph = sns.boxplot(x='Idade', y='QTDEPGTO', data=dataset, orient="h")

dataset = dataset.drop(columns=['QTDEPGTO_FX'])

dataset[['QTDEPGTO']]

#tratamento da qtde pgto por faixas
for x in dataset['QTDEPGTO']:
  if np.isnan(x):
    dataset['QTDEPGTO_FX'] = 'Nulo'
  elif x <= 4.0:
    dataset['QTDEPGTO_FX'] = '1 a 4'
  elif x <= 8.0:
    dataset['QTDEPGTO_FX'] = '5 a 8'
  elif x <= 12.0:
    dataset['QTDEPGTO_FX'] = '9 a 12'
  elif x <= 16.0:
    dataset['QTDEPGTO_FX'] = '13 a 16'
  else:
    dataset['QTDEPGTO_FX'] = 'Acima de 17'


#dataset['PRE_DIAS_P'] = [1 if np.isnan(x) or x > 60 else x/60 for x in dataset['DIAS_PRIMEIRA_PARCELA']]

print(dataset.head(100))

dataset['QTDEPGTO_FX'].value_counts()