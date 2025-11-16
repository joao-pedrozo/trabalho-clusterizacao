from pickle import load
import pandas as pd
import numpy as np

# =====================================
# 1️⃣ Carregar modelos
# =====================================
clusters = load(open('cluster_heart_failure.model', 'rb'))
normalizador_num = load(open('modelo_normalizador_heart_failure.model', 'rb'))

# =====================================
# 2️⃣ Definir colunas
# =====================================
colunas = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia_0', 'anaemia_1',
    'diabetes_0', 'diabetes_1',
    'high_blood_pressure_0', 'high_blood_pressure_1',
    'sex_0', 'sex_1',
    'smoking_0', 'smoking_1'
]

colunas_num = colunas[:7]
colunas_dummies = colunas[7:]

# =====================================
# 3️⃣ Montar DataFrame dos centroides
# =====================================
df = pd.DataFrame(clusters.cluster_centers_, columns=colunas)

# Reverter normalização
df[colunas_num] = normalizador_num.inverse_transform(df[colunas_num])

# =====================================
# 4️⃣ Reverter dummies (argmax + from_dummies)
# =====================================
# Agrupar dummies pelo prefixo antes do "_"
grupos = {}
for c in colunas_dummies:
    prefix = c.split('_', 1)[0]
    grupos.setdefault(prefix, []).append(c)

# Criar DataFrame binário (0/1) com base no maior valor por grupo
df_bin = pd.DataFrame(0, index=df.index, columns=colunas_dummies)
for prefix, cols in grupos.items():
    idx_max = df[cols].values.argmax(axis=1)
    df_bin.loc[:, cols] = 0
    df_bin.values[np.arange(len(df)), [colunas_dummies.index(cols[i]) for i in idx_max]] = 1

# Reverter one-hot → categorias originais
df_cat = pd.from_dummies(df_bin, sep='_')

# =====================================
# 5️⃣ Juntar numéricas + categóricas
# =====================================
df_final = pd.concat([df[colunas_num], df_cat], axis=1)

# =====================================
# 6️⃣ Exibir resultados
# =====================================
pd.set_option('display.max_columns', None)

print("\n===== CENTROIDES (reconstruídos) =====\n")
print(df_final.round(3))

print("\n===== DESCRIÇÃO LEGÍVEL POR CLUSTER =====\n")
for i, linha in df_final.iterrows():
    print(f"--- Cluster {i} ---")
    print("Numéricas:", {k: round(linha[k], 2) for k in colunas_num})
    print("Categóricas:")
    for c in df_cat.columns:
        print(f"  {c}: {linha[c]}")
    print()

