from pickle import load
import pandas as pd
import numpy as np

# ---------------------------
# 1) Carregar modelos
# ---------------------------
clusters = load(open('cluster_heart_failure.model', 'rb'))
normalizador_num = load(open('modelo_normalizador_heart_failure.model', 'rb'))

# ---------------------------
# 2) Colunas (mesma ordem do treino)
# ---------------------------
colunas = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia_0', 'anaemia_1',
    'diabetes_0', 'diabetes_1',
    'high_blood_pressure_0', 'high_blood_pressure_1',
    'sex_0', 'sex_1',
    'smoking_0', 'smoking_1'
]

colunas_numericas = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'time']
dummy_cols = [c for c in colunas if c not in colunas_numericas]

# ---------------------------
# 3) DataFrame dos centroides
# ---------------------------
df_centroides = pd.DataFrame(clusters.cluster_centers_, columns=colunas)

# ---------------------------
# 4) Reverter normalização numérica
# ---------------------------
df_centroides[colunas_numericas] = normalizador_num.inverse_transform(
    df_centroides[colunas_numericas]
)

# ---------------------------
# 5) Converter dummies contínuos em dummies binárias (one-hot por grupo)
# ---------------------------
# Agrupar colunas dummy por prefixo (tudo antes do primeiro '_')
groups = {}
for col in dummy_cols:
    prefix = col.split('_', 1)[0]
    groups.setdefault(prefix, []).append(col)

# DataFrame base contendo os valores originais (contínuos) das dummies
df_dummies_cont = df_centroides[dummy_cols].copy()

# Criar DataFrame binário (zeros) com mesmas colunas
df_dummies_bin = pd.DataFrame(0, index=df_dummies_cont.index, columns=df_dummies_cont.columns)

# Para cada grupo (ex: anaemia, diabetes, high_blood_pressure, ...)
for prefix, cols in groups.items():
    # argmax ao longo das colunas do grupo para cada linha -> escolha da categoria dominante
    chosen_indices = df_dummies_cont[cols].values.argmax(axis=1)
    for row_idx, col_idx in enumerate(chosen_indices):
        chosen_col = cols[col_idx]
        df_dummies_bin.at[row_idx, chosen_col] = 1

# ---------------------------
# 6) Reverter o from_dummies (apenas nas dummies binarias)
# ---------------------------
# pd.from_dummies() espera colunas 0/1 ou bool-like e converte para colunas categóricas
df_categorias = pd.from_dummies(df_dummies_bin, sep='_')

# ---------------------------
# 7) Montar DataFrame final
# ---------------------------
df_final = pd.concat([df_centroides[colunas_numericas].reset_index(drop=True),
                      df_categorias.reset_index(drop=True)], axis=1)

# ---------------------------
# 8) Impressão / descrição dos centroides
# ---------------------------
pd.set_option('display.max_columns', None)
print("\n===== CENTROIDES (reconstruídos) =====\n")
print(df_final.round(3))

print("\n===== DESCRIÇÃO LEGÍVEL POR CLUSTER =====\n")
for i, row in df_final.iterrows():
    print(f"--- Cluster {i} ---")
    # Numéricas com arredondamento
    num_dict = {k: float(np.round(row[k], 3)) for k in colunas_numericas}
    # Categóricas: pegar colunas que não são numéricas no df_final
    cat_cols = [c for c in df_final.columns if c not in colunas_numericas]
    cat_values = {c: row[c] for c in cat_cols}
    # Filtrar e transformar em pares categoria:valor (apenas para mostrar)
    # df_categorias tem colunas com nomes das categorias (ex: anaemia, diabetes, ...)
    # então exibir diretamente:
    print("Numéricas:", num_dict)
    print("Categóricas:")
    # se df_categorias tiver colunas categóricas (não dummies) — exibe valores
    for c in df_categorias.columns:
        print(f"  {c}: {row[c]}")
    print()

