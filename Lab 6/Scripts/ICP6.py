# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:17:02 2021

@author: Dietrich
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#%% Read Data

data = pd.read_csv('CC.csv')
x = data.drop(labels=['TENURE', 'CUST_ID'], axis=1).apply(lambda x: x.fillna(x.mean()))
y = data.iloc[:,-1]

#%% Elbow Method

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#%% Apply Feature Scaling

scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)


#%% Apply PCA

pca = PCA(2)
x_pca = pca.fit_transform(x)

#%% Apply K Means
nclusters = 3
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x_pca)
y_cluster_kmeans=km.predict(x_pca)
score=metrics.silhouette_score(x_pca,y_cluster_kmeans)
print("Score is: ", score)

