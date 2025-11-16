from pickle import load
import pandas as pd
import numpy as np

# ==========================================
# 1) Carregar modelos
# ==========================================
clusters = load(open('cluster_heart_failure.model', 'rb'))
normalizador_num = load(open('modelo_normalizador_heart_failure.model', 'rb'))

# ==========================================
# 2) Colunas e estrutura original
# ==========================================
colunas = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'
]

# ==========================================
# 3) Entrar dados de um novo paciente
# ==========================================
# (poderia vir de input(), API ou arquivo CSV)
paciente = {
    'age': 65,
    'creatinine_phosphokinase': 582,
    'ejection_fraction': 20,
    'platelets': 265000,
    'serum_creatinine': 1.9,
    'serum_sodium': 130,
    'time': 4,
    'anaemia': 0,
    'diabetes': 0,
    'high_blood_pressure': 1,
    'sex': 1,
    'smoking': 0
}

# Converter para DataFrame
df_paciente = pd.DataFrame([paciente])

# Separar colunas numéricas e binárias
colunas_binarias = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
# Converter colunas binárias para string (igual ao treino)
df_paciente[colunas_binarias] = df_paciente[colunas_binarias].astype(str)

# ==========================================
# 4) Pré-processamento igual ao treino
# ==========================================
# Aplicar get_dummies para gerar mesmas colunas
df_paciente_dummies = pd.get_dummies(df_paciente, prefix=colunas_binarias, dtype=int)

# Garantir que todas as colunas esperadas existam
colunas_treino = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia_0', 'anaemia_1',
    'diabetes_0', 'diabetes_1',
    'high_blood_pressure_0', 'high_blood_pressure_1',
    'sex_0', 'sex_1',
    'smoking_0', 'smoking_1'
]

# Adicionar colunas ausentes com 0
for col in colunas_treino:
    if col not in df_paciente_dummies.columns:
        df_paciente_dummies[col] = 0

# Reordenar colunas
df_paciente_dummies = df_paciente_dummies[colunas_treino]

# Normalizar as colunas numéricas
colunas_num = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
               'serum_creatinine', 'serum_sodium', 'time']
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
for c in colunas_centroides[7:]:
    # Para "high_blood_pressure_0", queremos "high_blood_pressure"
    # Para "anaemia_1", queremos "anaemia"
    parts = c.rsplit('_', 1)  # Split do último underscore
    prefix = parts[0]
    grupos.setdefault(prefix, []).append(c)

df_bin = pd.DataFrame(0, index=df_centroides.index, columns=colunas_centroides[7:])
for prefix, cols in grupos.items():
    idx_max = df_centroides[cols].values.argmax(axis=1)
    df_bin.loc[:, cols] = 0
    df_bin.values[np.arange(len(df_centroides)), [colunas_centroides[7:].index(cols[i]) for i in idx_max]] = 1

# Reverter one-hot → categorias originais manualmente
# (não podemos usar from_dummies porque high_blood_pressure tem múltiplos underscores)
df_cat = pd.DataFrame(index=df_bin.index)
for prefix, cols in grupos.items():
    # Para cada grupo, pegar o valor (0 ou 1) que está no final do nome da coluna
    valores = []
    for idx in df_bin.index:
        # Encontrar qual coluna tem valor 1 para este prefixo
        valor_encontrado = None
        for col in cols:
            if df_bin.loc[idx, col] == 1:
                # Extrair o valor do final (0 ou 1)
                valor = col.rsplit('_', 1)[1]
                valor_encontrado = int(valor)
                break
        valores.append(valor_encontrado if valor_encontrado is not None else 0)
    df_cat[prefix] = valores
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

