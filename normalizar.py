#Normalizar dados
#Instalar
# Pandas, scikit-learn, numpy

#Libs para normalização
from sklearn import preprocessing
from pickle import dump #Para salvar

import pandas as pd

#Abrir o arquivo de dados
#***


dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', sep = ',')

#excluir a coluna NObeyesdad exclusivamente para a clusterização
dados = dados.drop(columns=['NObeyesdad'])

#Separar as colunas numericas as categóricas

#Objeto com as colunas numericas
dados_num = dados.drop(columns=[
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'])

#Objeto de dados somente com as colunas categóricas
dados_cat = dados[[
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS']]

#Construir o objeto normalizador numérico (MinMaxScaler)
normalizador_numerico = preprocessing.MinMaxScaler()

#Treinar o modelo normalizador
modelo_normalizador = normalizador_numerico.fit(dados_num)
#Salvar o modelo para uso posterior
dump(modelo_normalizador, open('modelo_normalizador_obesity.model', 'wb'))

#########################
# Normalizar os dados
# Normalização numérica
dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)

#Normalização categórica
#***
dados_cat_normalizados = pd.get_dummies(data=dados_cat, prefix=[
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'],
     dtype=int)


#Recompor os dados, unindo os numéricos e os categóricos
#Transformar os dados numericos normalizados em um dataframe do Pandas
dados_num = pd.DataFrame(data=dados_num_normalizados, columns=dados_num.columns)

#Juntar o dados_num (já normalizado) com o dados_cat_normalizados
dados = dados_num.join(dados_cat_normalizados)

#***
dados.to_csv('dados_preprocessados_obesidade.csv', index=False)