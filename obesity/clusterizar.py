from sklearn import preprocessing
from pickle import dump #Para salvar

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

#Determinar o número ideal de grupos a serem obtidos
#Método do cotovelo (Elbow) pelas distorções
from sklearn.cluster import KMeans #Método de clusterização
from scipy.spatial.distance import cdist #para avaliar as distancias e as distorcoes

#Carregar os dados preprocessados
dados = pd.read_csv('dados_preprocessados_obesidade.csv', sep = ',')

#Matriz para armazenar as distorcoes
distorcoes = []

#Intervalo para testagem
K = range(2, 50) #Intervalo de 2 a 49
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados,
                      cluster_model.cluster_centers_,
                      'euclidean'), axis=1)/dados.shape[0]
            )
        )
#Gerar o gráfico das distorcoes
fig, ax = plt.subplots()
ax.plot(K, distorcoes)
ax.set(xlabel='n Clusters', ylabel='Distorcoes')
ax.grid()
plt.show()
# plt.savefig('distorcoes.jpg')    

#Determinar o número ideal de clusters
#Método do cotovelo (elbow) a partir das distorcoes
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1] #Retorna o último elemento da matriz K
yn = distorcoes[-1] #Retorna o último elemento da matriz distorcoes
distancias = [] #matriz para as distâncias
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)
numero_clusters_otimo = K[
                            distancias.index(
                                np.max(distancias)
                                )
                        ]
print('Número ótimo de clustes:', numero_clusters_otimo)

#Treinar o modelo de clustes com o número otimizado
cluster_obesity = KMeans(n_clusters=numero_clusters_otimo).fit(dados)
#Salvar o modelo de clusters em disco
dump(cluster_obesity, open('cluster_obesity.model', 'wb'))
