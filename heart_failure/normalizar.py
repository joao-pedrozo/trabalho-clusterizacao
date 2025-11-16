#Normalizar dados
#Instalar
# Pandas, scikit-learn, numpy

#Libs para normalização
from sklearn import preprocessing
from pickle import dump #Para salvar

import pandas as pd

#Abrir o arquivo de dados
dados = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep = ',')

#excluir a coluna DEATH_EVENT exclusivamente para a clusterização
dados = dados.drop(columns=['DEATH_EVENT'])

#Separar as colunas numericas as categóricas

#Objeto com as colunas numericas
dados_num = dados.drop(columns=[
    'anaemia',
    'diabetes',
    'high_blood_pressure',
    'sex',
    'smoking'])

#Objeto de dados somente com as colunas categóricas
dados_cat = dados[[
    'anaemia',
    'diabetes',
    'high_blood_pressure',
    'sex',
    'smoking']]

#Construir o objeto normalizador numérico (MinMaxScaler)
normalizador_numerico = preprocessing.MinMaxScaler()

#Treinar o modelo normalizador
modelo_normalizador = normalizador_numerico.fit(dados_num)
#Salvar o modelo para uso posterior
dump(modelo_normalizador, open('modelo_normalizador_heart_failure.model', 'wb'))

#########################
# Normalizar os dados
# Normalização numérica
dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)

#Normalização categórica
#***
dados_cat_normalizados = pd.get_dummies(data=dados_cat, prefix=[
    'anaemia',
    'diabetes',
    'high_blood_pressure',
    'sex',
    'smoking'],
     dtype=int)


#Recompor os dados, unindo os numéricos e os categóricos
#Transformar os dados numericos normalizados em um dataframe do Pandas
dados_num = pd.DataFrame(data=dados_num_normalizados, columns=dados_num.columns)

#Juntar o dados_num (já normalizado) com o dados_cat_normalizados
dados = dados_num.join(dados_cat_normalizados)

# Garantir ordem das colunas (numéricas primeiro, depois dummies)
colunas_ordenadas = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia_0', 'anaemia_1',
    'diabetes_0', 'diabetes_1',
    'high_blood_pressure_0', 'high_blood_pressure_1',
    'sex_0', 'sex_1',
    'smoking_0', 'smoking_1'
]
# Reordenar apenas as colunas que existem
dados = dados[[col for col in colunas_ordenadas if col in dados.columns]]

#***
dados.to_csv('dados_preprocessados_heart_failure.csv', index=False)

