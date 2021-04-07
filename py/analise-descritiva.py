# @title Importar as bibliotecas e carregar os dados.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import numpy as np

dataset = pd.read_csv('/content/AULA_BASE_FULL.txt', sep='\t') # Separador TAB
dataset.head(5)

dataset['Sexo'].value_counts()

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





""""
OK 3   Sexo                     	  	4986 non-null   object 
4   DataNascimento         	  	5000 non-null   object 
6   Uf                       	  	1876 non-null   object 
OK 7   EstadoCivil         	  	4990 non-null   object 
OK 8   Profissao               	  	4669 non-null   object 
OK 10  Carteira               	  	4775 non-null   object 
11  DataRecebimentoContrato  	4775 non-null   object 
OK 12  VencDebito              	  	4560 non-null   object 
13  DataAssociacao         	  	4775 non-null   object 
14  qtdeTItulo               	 	4412 non-null   float64
OK 15  VLR_DEBITO             		4412 non-null   float64
16  QTDE_TENTATIVAS_CTTO     	4652 non-null   float64
OK 17  QTDEPGTO                 	548 non-null    float64
OK 18  VLR_PGTO                 	 	548 non-null    float64
OK 19 Idade 
OK 20 IdadeDebito"""

dataset = dataset.drop(columns=['QTDEPGTO_FX'])

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

dataset['VLR_PGTO'].value_counts()