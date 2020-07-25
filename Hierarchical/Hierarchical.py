import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#data set ra vared mikonim
dataset=pd.read_csv(r"dataset.csv")
X=dataset.iloc[:,[3,4]].values

#az dendogram to estefade mikonim ta optimal teedad cluster ha ra darbiavarim
#nemoodar ra mikeshim
dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.ylabel("Euclidian Distane")
plt.xlabel("clusters")
plt.show()

#hc ra ba tabe clustering ii ke bala vared kardim be dataset fit mikonim
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(X)

#cluster ra mikeshim
#ba k means
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="green",label="cluster 2")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c="blue",label="cluster 3")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c="orange",label="cluster 4")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c="violet",label="cluster 5")
plt.legend()
plt.title("K-MEANS")
plt.xlabel("Income")
plt.ylabel('Score')
plt.show()
