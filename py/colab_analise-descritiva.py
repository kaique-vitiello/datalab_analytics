# -*- coding: utf-8 -*-

# @title Importar as bibliotecas e carregar os dados.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import numpy as np

dataset = pd.read_csv('/content/AULA_BASE_FULL.txt', sep='\t') # Separador TAB
dataset.head(5)

#@title Avaliar a quantidade de dados preenchidos em cada variavel.
dataset.info()

#@title Analise de missings por variaveis
dataset.isnull().sum()

# @title Converte a data de nascimento para "datetime" depois, calcula a idade com base na data de nascimento e o dia de hoje, em seguida cria uma nova coluna com a idade de cada pessoa.
dataset['DataNascimento'] = pd.to_datetime(dataset['DataNascimento'])
dataset['Idade'] = (datetime.now() - dataset['DataNascimento']) / 365
dataset['Idade'] = (dataset['Idade']).dt.days
dataset['Idade'].value_counts()

# @title Converte a data de vencimento do debito para "datetime" depois, calcula a idade do debito.
dataset['VencDebito'] = pd.to_datetime(dataset['VencDebito'])
dataset['IdadeDebito'] = (datetime.now() - dataset['VencDebito']) / 365
dataset['IdadeDebito'] = (dataset['IdadeDebito']).dt.days
dataset['IdadeDebito'].value_counts()

#@title Avaliar as variáveis quantitativas (Avg, Desvio Padrão, Max e Min).
dataset.describe().round()

#@title Analise de outliers na variavel VLR_PGTO x Sexo.
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(8,6))

graph = sns.boxplot(x='Sexo', y='VLR_PGTO', data=dataset, orient="v")

#@title Eliminamos dados nulos e validamos que o maior volume de contratos está concentrado no valor de até R$2000,00.

filter_data = dataset.dropna(subset=['VLR_PGTO'])
plt.figure(figsize=(10,8))
ax = sns.histplot(filter_data['VLR_PGTO'], color='g', kde=False)

#@title Analisamos a frequência relativa por sexo dos devedores.

filter_data = dataset.dropna(subset=['Sexo'])
type_counts = filter_data['Sexo'].value_counts()

dataset2 = pd.DataFrame({'Sexo': type_counts}, index = ['M', 'F'])
dataset2.plot.pie(y='Sexo', figsize=(8,8), autopct='%1.1f%%')

#@title Analisamos a quantidade de devedores por estado Civil, podemos concluir que Casado x Solteiro estão com valores bem proximos.

plt.figure(figsize=(10,5))
graph = sns.countplot(x='EstadoCivil', data=dataset)

#@title Analisamos a quantidade de devedores por Estado.

plt.figure(figsize=(20,10))
graph = sns.countplot(x='Uf', data=dataset)

#@title Analisamos a quantidade de devedores por tempo de debito.

plt.figure(figsize=(10,5))
graph = sns.countplot(x='IdadeDebito', data=dataset)

#@title Analisamos a quantidade de devedores por Carteira
plt.figure(figsize=(10,5))
graph = sns.countplot(x='Carteira', data=dataset)

#@title Mostra em histograma a quantidade geral por idade
filter_data = dataset.dropna(subset=['Idade'])
plt.figure(figsize=(10,8))
ax = sns.histplot(filter_data['Idade'], color='g', kde=True)

#@title Analise de outliers nas variaveis Carteira x Idade
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(8,6))

graph = sns.boxplot(x='Carteira', y='Idade', data=dataset, orient="v")

#@title Analise de outliers nas variaveis QTDPGTO x Sexo
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(10,8))
graph = sns.boxplot(x='QTDEPGTO', y='Sexo', data=dataset, orient="h")

#@title Analise de outliers nas variaveis IdadeDebito x Idade
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")
plt.figure(figsize=(8,6))

graph = sns.boxplot(x='IdadeDebito', y='Idade', data=dataset, orient="v")