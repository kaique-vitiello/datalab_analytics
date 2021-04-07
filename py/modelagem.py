# Bibliotecas padrão
import numpy as np
import pandas as pd
from datetime import datetime
from math import sqrt

## Carregando os dados
dataset = pd.read_csv('/content/drive/MyDrive/DataLab/AULA_BASE_FULL.txt',sep='\t') # Separador TAB

# @title Converte a data de nascimento para "datetime" depois, calcula a idade com base na data de nascimento e o dia de hoje, em seguida cria uma nova coluna com a idade de cada pessoa.
dataset['DataNascimento'] = pd.to_datetime(dataset['DataNascimento'])
dataset['Idade'] = (datetime.now() - dataset['DataNascimento']) / 365
dataset['Idade'] = (dataset['Idade']).dt.days

# @title Calcula a Idade no vencimento do Debito
dataset['VencDebito'] = pd.to_datetime(dataset['VencDebito'])
dataset['IdadeVencDebito'] = (dataset['VencDebito'] - dataset['DataNascimento']) / 365
dataset['IdadeVencDebito'] = (dataset['IdadeVencDebito']).dt.days

# @title Calcula a Idade no Recebimento do Contrato
dataset['DataRecebimentoContrato'] = pd.to_datetime(dataset['DataRecebimentoContrato'])
dataset['IdadeRecebContr'] = (dataset['DataRecebimentoContrato'] - dataset['DataNascimento']) / 365
dataset['IdadeRecebContr'] = (dataset['IdadeRecebContr']).dt.days

# Pré-processamento das variáveis
dataset['PRE_SEXO_M'] = [1 if x=='M' else 0 for x in dataset['Sexo']]
dataset['PRE_SEXO_F'] = [1 if x=='F' else 0 for x in dataset['Sexo']]
dataset['PRE_ESTCIV_CAS_1'] = [1 if x=='Casado(a)' or x=='CASADO' else 0 for x in dataset['EstadoCivil']]
dataset['PRE_ESTCIV_CAS_2'] = [1 if x=='Solteiro(a)' or x=='SOLTEIRO' else 0 for x in dataset['EstadoCivil']]
dataset['PRE_ESTCIV_CAS_3'] = [1 if x=='Outro' or x=='OUTROS' else 0 for x in dataset['EstadoCivil']]
dataset['PRE_CART_1'] = [1 if x=='PDD (121 a 360)' else 0 for x in dataset['Carteira']]
dataset['PRE_CART_2'] = [1 if x=='FASES 1 (361 a 720)' else 0 for x in dataset['Carteira']]
dataset['PRE_CART_3'] = [1 if x=='FASES 2 (721 a 1800)' else 0 for x in dataset['Carteira']]
dataset['PRE_UF_1'] = [1 if x=='MT' else 0 for x in dataset['Uf']]
dataset['PRE_UF_2'] = [1 if x=='MT'or x=='GO' or x=='PR' or x=='PR' or x=='SP' or x=='SC' or x=='MG' else 0 for x in dataset['Uf']]
dataset['PRE_UF_3'] = [1 if x=='AL'or x=='BA' or x=='CE' or x=='DF' or x=='MA' or x=='PA' or x=='PB' or x=='PE' or x=='PE' or x=='RJ' or x=='RN' else 0 for x in dataset['Uf']]
# Idade pessoa (Normalização)
dataset['PRE_IDADE'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['Idade']] 
dataset['PRE_IDADE'] = [85 if x > 85 else x for x in dataset['PRE_IDADE']] 
dataset['PRE_IDADE'] = [(x-18)/(85-18) for x in dataset['PRE_IDADE']] 
# Idade Venc Debito (Normalização)
dataset['PRE_IDADE_VEN_DBT'] = [0 if np.isnan(x) else x for x in dataset['IdadeVencDebito']] 
dataset['PRE_IDADE_VEN_DBT'] = [x/(dataset['IdadeVencDebito'].max()) for x in dataset['PRE_IDADE_VEN_DBT']] 
# Idade Receb Contr (Normalização)
dataset['PRE_IDADE_RCB_CONTR'] = [0 if np.isnan(x) else x for x in dataset['IdadeRecebContr']] 
dataset['PRE_IDADE_RCB_CONTR'] = [x/(dataset['IdadeRecebContr'].max()) for x in dataset['PRE_IDADE_RCB_CONTR']]
# Qtd de titulos (Normalização)
dataset['PRE_QTDTIT'] = [0 if np.isnan(x) else x for x in dataset['qtdeTItulo']]
dataset['PRE_QTDTIT'] = [30 if x > 30 else x for x in dataset['PRE_QTDTIT']]
dataset['PRE_QTDTIT'] = [x/30 for x in dataset['PRE_QTDTIT']]
# Valor do debito (Normalização)
dataset['PRE_VALOR_DEB'] = [0 if np.isnan(x) else x for x in dataset['VLR_DEBITO']]
dataset['PRE_VALOR_DEB'] = [x/(dataset['VLR_DEBITO'].max()) for x in dataset['PRE_VALOR_DEB']] 
# Target
dataset['TARGET_NUM'] = [1 if x>=1 else 0 for x in dataset['QTDEPGTO']]
dataset['TARGET_DISC'] = ['BOM' if x>=1 else 'MAU' for x in dataset['QTDEPGTO']]

cols_in = ['PRE_SEXO_M',
           'PRE_SEXO_F',
           'PRE_ESTCIV_CAS_1',
           'PRE_ESTCIV_CAS_2',
           'PRE_ESTCIV_CAS_3',
           'PRE_CART_1',
           'PRE_CART_2',
           'PRE_CART_3',
           'PRE_UF_1',
           'PRE_UF_2',
           'PRE_UF_3',
           'PRE_IDADE',
           'PRE_IDADE_VEN_DBT',
           'PRE_IDADE_RCB_CONTR',
           'PRE_QTDTIT',
           'PRE_VALOR_DEB'
           ]    

target = 'TARGET_NUM'

dataset[cols_in]

# Exportando os dados pre-processados - Google Drive
dataset.to_csv('/content/drive/MyDrive/DataLab/resultado_preproc.csv')

# Separando em dados de treinamento e teste
y = dataset[target]
X = dataset[cols_in]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size = 0.3, random_state = 126)

#Importar modelos na biblioteca SKLEARN
import xgboost
from xgboost import XGBClassifier
import sklearn
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet, LinearRegression, RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier

# @title Selecionando Atributos com RFE - Recursive Feature Elimination 
model = LogisticRegression(solver='newton-cg')
selected = RFE(model,step=1,n_features_to_select=12).fit(X_train, y_train)

print('------------------------ SELEÇÃO DE VARIÁVEIS --------------------------------------')
print("Num Features: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas por RFE
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas por RFE

## Ajustando e executando os modelos - Aprendizado supervisionado 

# Regressão linear com dados de treinamento
LinearReg = LinearRegression(fit_intercept=True)
LinearReg.fit(X_train, y_train)

# Regressao logistica com dados de treinamento
LogisticReg = LogisticRegression()
LogisticReg.fit(X_train, y_train)

# Árvore de decisão com dados de treinamento
dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=30, min_samples_split=30,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=0, splitter='best')
dtree.fit(X_train, y_train)

#Rede Neural com dados de treinamento
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(25), learning_rate='constant',
       learning_rate_init=0.01, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.25, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train)

#XGBClassifier
XGBoost = xgboost.XGBClassifier(learning_rate =0.05, n_estimators=100, max_depth=1,
             min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
             objective= 'binary:logistic', booster = 'gbtree', nthread=4, scale_pos_weight=1,
             seed=27, random_state=1337, num_boost_round = 999)
XGBoost.fit(X_train, y_train)

# @title Previsão treinamento e teste - CLASSIFICAÇÃO

# Regressão Linear
y_pred_train_RL = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_train)] )
y_pred_test_RL  = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_test)])

# Regressão Logística
y_pred_train_RLog = LogisticReg.predict(X_train)
y_pred_test_RLog  = LogisticReg.predict(X_test)

# Árvore de Decisão
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)

#XGBClassifier
y_pred_train_XGBoost = XGBoost.predict(X_train)
y_pred_test_XGBoost = XGBoost.predict(X_test)

## Cálcula e mostra a Acurácia dos modelos

print('Acurácia Regressão Linear:    ',metrics.accuracy_score(y_test, y_pred_test_RL))
print('Acurácia Regressão Logística: ',metrics.accuracy_score(y_test, y_pred_test_RLog))
print('Acurácia Árvore de Decisão:   ',metrics.accuracy_score(y_test, y_pred_test_DT))
print('Acurácia XGBoost:             ',metrics.accuracy_score(y_test, y_pred_test_XGBoost ))

# Matriz de confusão dos modelos

print()
print('----------     MATRIZ DE CONFUSÃO    ----------------')
print()
print('--  Árvore de Decisão  --')
print()
print(pd.crosstab(y_test, y_pred_test_DT, rownames=['Real'], colnames=['Predito'], margins=True))
print()
print('--  Regressão Linear  --')
print()
print(pd.crosstab(y_test, y_pred_test_RL, rownames=['Real'], colnames=['Predito'], margins=True))
print()
print('--  Regressão Logística  --')
print()
print(pd.crosstab(y_test, y_pred_test_RLog, rownames=['Real'], colnames=['Predito'], margins=True))
print()
print()
print('--  XGBoost  --')
print()
print(pd.crosstab(y_test, y_pred_test_XGBoost, rownames=['Real'], colnames=['Predito'], margins=True))
print()
print()
print(y_pred_test_RNA)

# @title Previsão treinamento e teste - REGRESSÃO

# Regressão Linear
y_pred_train_RL_R = LinearReg.predict(X_train)
y_pred_test_RL_R  = LinearReg.predict(X_test)

# Regressão Logística
y_pred_train_RLog_R = LogisticReg.predict_proba(X_train)[:,1]
y_pred_test_RLog_R  = LogisticReg.predict_proba(X_test)[:,1]

# Árvore de Decisão
y_pred_train_DT_R  = dtree.predict_proba(X_train)[:,1]
y_pred_test_DT_R  = dtree.predict_proba(X_test)[:,1]

# XGBoost
y_pred_train_XGBoost_R = XGBoost.predict_proba(X_train)[:,1]
y_pred_test_XGBoost_R = XGBoost.predict_proba(X_test)[:,1]

print('----------------     ACURÁCIA     ------------------------------------------')
## Cálcula e mostra a Acurácia dos modelos
print('Acurácia Regressão Linear:    ',metrics.accuracy_score(y_test, y_pred_test_RL))
print('Acurácia Regressão Logística: ',metrics.accuracy_score(y_test, y_pred_test_RLog))
print('Acurácia Árvore de Decisão:   ',metrics.accuracy_score(y_test, y_pred_test_DT))
print('Acurácia XGBoost:             ',metrics.accuracy_score(y_test, y_pred_test_XGBoost))
print()
print()

## Cálcula e mostra RMSE dos modelos
print('----------------     RMSE ERROR    -----------------------------------------')
print('Regressão Linear:   ',  sqrt(np.mean((y_pred_test_RL_R -  y_test) ** 2) ))
print('Regressão Logística:',  np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5)
print('Árvore de Decisão:  ',  sqrt(np.mean((y_test - y_pred_test_DT_R) **2) ))
print('XGBoost:            ',  sqrt(np.mean((y_test - y_pred_test_XGBoost_R) **2) ))
print()
print()

## Cálcula o KS2
print('----------------     KS2    ------------------------------------------------')
#função para calculo de KS2
from scipy.stats import ks_2samp
def KS2(y, y_pred):
    df_ks2 = pd.DataFrame([x for x in y_pred], columns=['REGRESSION_RLog'])
    df_ks2['ALVO'] = [x for x in y]
    return ks_2samp(df_ks2.loc[df_ks2.ALVO==0,"REGRESSION_RLog"], df_ks2.loc[df_ks2.ALVO==1,"REGRESSION_RLog"])[0]

print('Regressão Linear:    ', KS2(y_test,y_pred_test_RL_R))
print('Regressão Logística: ', KS2(y_test,y_pred_test_RLog_R))
print('Árvore de Decisão:   ', KS2(y_test,y_pred_test_DT_R))
print('XGBoost:             ', KS2(y_test,y_pred_test_XGBoost_R))

#@title Montando um Data Frame com os resultados e exportando para o Google Drive.

# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT_R, columns=['REGRESSION_DT'])
df_train['CLASSIF_DT'] = y_pred_train_DT
df_train['REGRESSION_RL'] = y_pred_train_RL_R
df_train['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_train_RL]
df_train['REGRESSION_RLog'] = y_pred_train_RLog_R
df_train['CLASSIF_RLog'] = y_pred_train_RLog
df_train['REGRESSION_XGBoost'] = y_pred_train_XGBoost_R
df_train['CLASSIF_XGBoost'] = y_pred_train_XGBoost
df_train['ALVO'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'
df_train['Contrato'] = dataset['Contrato']

# Conjunto de teste
df_test = pd.DataFrame(y_pred_test_DT_R, columns=['REGRESSION_DT'])
df_test['CLASSIF_DT'] = y_pred_test_DT
df_test['REGRESSION_RL'] = y_pred_test_RL_R
df_test['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_test_RL]
df_test['REGRESSION_RLog'] = y_pred_test_RLog_R
df_test['CLASSIF_RLog'] = y_pred_test_RLog
df_test['REGRESSION_XGBoost'] = y_pred_test_XGBoost_R
df_test['CLASSIF_XGBoost'] = y_pred_test_XGBoost
df_test['ALVO'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 
df_test['Contrato'] = dataset['Contrato']

print()
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train], sort = False)

## Exportando os dados para avaliação dos resultados em outra ferramenta
df_total.to_csv('/content/drive/MyDrive/DataLab/resultado_comparacao2.csv')
print('----------------     RESULTADOS EXPORTADOS PARA O GOOGLE DRIVE   ------------------------------------')