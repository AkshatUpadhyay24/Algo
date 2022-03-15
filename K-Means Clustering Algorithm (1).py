#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('income.csv')
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[4]:


df.shape


# In[5]:


plt.xlabel('age')
plt.ylabel('income')
plt.scatter(df.age,df.income,color='red',marker="+")


# In[6]:


# now we create K-means objects of 3 clusters

km=KMeans(n_clusters=3)


# In[7]:


# In K-mean cluster fit and predict will b apply at same time.

y_pred=km.fit_predict(df[['age','income']])
y_pred


# In[8]:


# now we put 'y_pred' values in 'cluster' column in data frame.

df['cluster']=y_pred
df.head()


# In[9]:


# now will find "centroids" of 3 clusters.

km.cluster_centers_


# In[10]:


# now we have to make 3 data frames of diff-2 clusters to make visualization easy.

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]


# In[11]:


# now will plot scatter plot "age" vs "income".

plt.xlabel('age')
plt.ylabel('income')
plt.scatter(df1.age,df1.income,color='red',marker='+')
plt.scatter(df2.age,df2.income,color='blue',marker='*')
plt.scatter(df3.age,df3.income,color='black',marker='^')
plt.legend()


# In[12]:


# when scale is not good during visualization, then will do 'Normalization' of age and income through --> 'MinMaxScaler'.

scaler=MinMaxScaler()

scaler.fit(df[['income']])
df['income']=scaler.transform(df[['income']])

scaler.fit(df[['age']])
df['age']=scaler.transform(df[['age']])

df.head()


# In[13]:


# In K-mean cluster fit and predict will b apply at same time again.


km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df[['age','income']])
y_pred


# In[14]:


# now we put 'y_pred' values in 'cluster' column in data frame again.

df['cluster']=y_pred
df.head()


# In[15]:


# now will find "centroids" of 3 clusters again.

km.cluster_centers_


# In[17]:


# To know the centers are set this tyme or not we need to visualize again.


plt.xlabel('age')
plt.ylabel('income')
plt.scatter(df1.age,df1['income'],color='red')
plt.scatter(df2.age,df2['income'],color='blue')
plt.scatter(df3.age,df3['income'],color='black')


# In[18]:


# jab zyada feature hote hai to pata nhi chalta ki 'K' kitna appropriate hona chahiye, uske liye "elbow" plot ka use kr skte h.

sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['age','income']])
    sse.append(km.inertia_)
    sse


# In[19]:


# Now we plot 2D graph of Elbow.

plt.xlabel('k')
plt.ylabel('Sum of Squared error')
plt.plot(k_rng,sse)


# In[ ]:


# So, Here "K=3" is the most Appropriate Value.

