#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import make_blobs, make_moons

X1, y1 = make_blobs(n_samples=1000, centers=4, cluster_std=0.5, n_features=2, random_state=0)
X2, y2 = make_blobs(n_samples=1000, centers=4, cluster_std=1, n_features=2, random_state=0)
X3, y3 = make_moons(n_samples=1000, noise=.05, random_state=0)


# In[5]:


sns.scatterplot(x = X1[:,0], y = X1[:,1], hue = y1)
plt.show()


# In[6]:


from sklearn.cluster import KMeans

# Especificamos el numero adecuado de clusters en cada caso
kmeans_1 = KMeans(n_clusters=4, random_state=0)
kmeans_2 = KMeans(n_clusters=4, random_state=0)
kmeans_3 = KMeans(n_clusters=2, random_state=0)


# In[7]:


kmeans_1.fit(X1)
kmeans_2.fit(X2)
kmeans_3.fit(X3)


# In[8]:


etiquetas_1 = kmeans_1.labels_
print(etiquetas_1.shape)


# In[9]:


centros_1 = kmeans_1.cluster_centers_
print(centros_1)


# In[10]:


etiquetas_2 = kmeans_2.labels_
centros_2 = kmeans_2.cluster_centers_
etiquetas_3 = kmeans_3.labels_
centros_3 = kmeans_3.cluster_centers_


# In[11]:


sns.scatterplot(X1[:, 0], X1[:, -1], hue = etiquetas_1)
sns.scatterplot(centros_1[:, 0], centros_1[:, 1],color='black', marker="+", s=1000)
plt.title('Data points and cluster centroids')
plt.show()


# In[ ]:




