from pickle import load
import pandas as pd
import numpy as np

# ==========================================
# 1) Carregar modelos
# ==========================================
clusters = load(open('cluster_obesity.model', 'rb'))
normalizador_num = load(open('modelo_normalizador_obesity.model', 'rb'))

# ==========================================
# 2) Colunas e estrutura original
# ==========================================
colunas = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
    'SCC', 'CALC', 'MTRANS'
]

# ==========================================
# 3) Entrar dados de um novo paciente
# ==========================================
# (poderia vir de input(), API ou arquivo CSV)
paciente = {
    'Age': 25,
    'Height': 1.75,
    'Weight': 70,
    'FCVC': 3,         # Frequency of vegetables consumption
    'NCP': 3,          # Number of main meals
    'CH2O': 2,         # Water intake
    'FAF': 2,          # Physical activity frequency
    'TUE': 1,          # Time using technology devices
    'Gender': 'Male',
    'family_history_with_overweight': 'yes',
    'FAVC': 'yes',
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'SCC': 'yes',
    'CALC': 'Sometimes',
    'MTRANS': 'Public_Transportation'
}

# Converter para DataFrame
df_paciente = pd.DataFrame([paciente])

# ==========================================
# 4) Pré-processamento igual ao treino
# ==========================================
# Aplicar get_dummies para gerar mesmas colunas
df_paciente_dummies = pd.get_dummies(df_paciente)

# Garantir que todas as colunas esperadas existam
colunas_treino = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender_Female', 'Gender_Male',
    'family_history_with_overweight_no', 'family_history_with_overweight_yes',
    'FAVC_no', 'FAVC_yes',
    'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'SMOKE_no', 'SMOKE_yes',
    'SCC_no', 'SCC_yes',
    'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
    'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

# Adicionar colunas ausentes com 0
for col in colunas_treino:
    if col not in df_paciente_dummies.columns:
        df_paciente_dummies[col] = 0

# Reordenar colunas
df_paciente_dummies = df_paciente_dummies[colunas_treino]

# Normalizar as colunas numéricas
colunas_num = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df_paciente_dummies[colunas_num] = normalizador_num.transform(df_paciente_dummies[colunas_num])

# ==========================================
# 5) Predizer o cluster
# ==========================================
cluster_predito = clusters.predict(df_paciente_dummies)[0]

# ==========================================
# 6) Carregar descrição dos centroides (revertidos)
# ==========================================
# (usa mesmo código simplificado da versão anterior)
colunas_centroides = colunas_treino
df_centroides = pd.DataFrame(clusters.cluster_centers_, columns=colunas_centroides)

# Reverter normalização
df_centroides[colunas_num] = normalizador_num.inverse_transform(df_centroides[colunas_num])

# Reverter dummies → categorias
grupos = {}
for c in colunas_centroides[8:]:
    prefix = c.split('_', 1)[0]
    grupos.setdefault(prefix, []).append(c)

df_bin = pd.DataFrame(0, index=df_centroides.index, columns=colunas_centroides[8:])
for prefix, cols in grupos.items():
    idx_max = df_centroides[cols].values.argmax(axis=1)
    df_bin.loc[:, cols] = 0
    df_bin.values[np.arange(len(df_centroides)), [colunas_centroides[8:].index(cols[i]) for i in idx_max]] = 1

df_cat = pd.from_dummies(df_bin, sep='_')
df_final = pd.concat([df_centroides[colunas_num], df_cat], axis=1)

# ==========================================
# 7) Exibir resultado
# ==========================================
print("\n===== DADOS DO PACIENTE =====")
for k, v in paciente.items():
    print(f"{k}: {v}")

print(f"\n O paciente pertence ao CLUSTER {cluster_predito}\n")

print("===== DESCRIÇÃO DO CLUSTER =====")
print(df_final.loc[cluster_predito].to_string())
