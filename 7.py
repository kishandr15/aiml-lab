from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("7_dataset.csv")
x1 = data['x'].values
x2 = data['y'].values
print(data)

X = np.matrix(list(zip(x1, x2)))

plt.scatter(x1, x2)
plt.show()

markers = ['s', 'o', 'v']
k = 3
cluster = KMeans(init='k-means++', n_clusters=k, n_init=10).fit(data)

for i, l in enumerate(cluster.labels_):
    plt.plot(x1[i], x2[i], marker=markers[l])



#  OUTPUT:

# Datasets:
# x,y
# 10,23
# 1,34
# 6,26
# 7,28
# 21,12
# 30,12
# 32,34
# 31,30
# 29,34
# 27,33
# 28,30
# 1,34
# 2,30
# 3,25
# 12,22
# 11,20
# 10,19
# 20,15
# 21,16
# 22,17
# 23,19