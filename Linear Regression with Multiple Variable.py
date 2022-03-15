#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[17]:


df=pd.read_csv('homeprice.csv')
df.head()


# In[18]:


df.fillna(df.bedroom.mean(),inplace=True)
df.mean()


# In[19]:


df.head()


# In[20]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[21]:


df1=df.drop(['price'],axis=1)
df1.head()


# In[22]:


df.price


# In[23]:


df.shape


# In[24]:


x=df1
y=df.price


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[26]:


model=linear_model.LinearRegression()
model


# In[27]:


model.fit(x,y)


# In[28]:


df.head()


# In[32]:


model.predict([[3000,4.0,15]])


# In[33]:


model.score(x,y)


# In[ ]:




