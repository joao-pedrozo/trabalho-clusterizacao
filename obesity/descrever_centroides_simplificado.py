from pickle import load
import pandas as pd
import numpy as np

# =====================================
# 1️⃣ Carregar modelos
# =====================================
clusters = load(open('cluster_obesity.model', 'rb'))
normalizador_num = load(open('modelo_normalizador_obesity.model', 'rb'))

# =====================================
# 2️⃣ Definir colunas
# =====================================
colunas = [
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

colunas_num = colunas[:8]
colunas_dummies = colunas[8:]

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
