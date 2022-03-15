#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PCA (Principal Component Analysis) Algorithm.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('heart.csv')
df.head()


# In[3]:


df.drop(['Unnamed: 0','chest pain type','resting ecg','ExerciseAngina','st_slope','sex'],axis=1,inplace=True)
df.head()


# In[4]:


df1=df.drop(['heart disease'],axis=1)
df1.head()


# In[36]:


df1.shape


# In[7]:


df.rename(columns={'resting bp':'resting_bp','fasting bs':'fasting_bs','max hr':'max_hr','old speak':'old_speak','heart disease':'heart_disease'},inplace=True)
df.head()


# In[9]:


x=df1
y=df.heart_disease


# In[11]:


# Now we do scaling first of 'x'

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[13]:


x_scaler=scaler.fit_transform(x)
x_scaler


# In[31]:


# before model training we hv to split data set in train & test.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


# In[32]:


# Now, for classification we use Logistic Regression.

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[35]:


# Now we use 'PCA' for fast computation

from sklearn.decomposition import PCA
pca=PCA(0.95) # 95% variation leni h
x_pca=pca.fit_transform(x)
x_pca.shape

# (120,6) ko (120,2) me divide kr diya.


# In[37]:


x_pca # show new numpy array.


# In[41]:


# New 2 columns of PCA component kitne variation capture karta hai, ye janne k liye will use--> "explained_variance_ratio_".
# means that much of % captured.

pca.explained_variance_ratio_


# In[42]:


pca.n_components_ # To know final Features


# In[45]:


# Now, we call train_test_split on new numpy array.

x_train_pca,x_test_pca,y_train,y_test=train_test_split(x_pca,y,test_size=0.2,random_state=30)


# In[46]:


# Again create Logistic Regression Model on new numpy array.

model=LogisticRegression(max_iter=1000) # zyada iter karke Gradient Descent ko converge kr diya.
model.fit(x_train_pca,y_train)
model.score(x_test_pca,y_test)


# In[70]:


# Now, sirf 2 PCA use krke dekhte hai, K Accuracy kitni aati hai.

pca=PCA(n_components=6)# if we increase components Accuracy will also increase.
x_pca=pca.fit_transform(x)
x_pca.shape


# In[71]:


pca.explained_variance_ratio_


# In[72]:


x_train_pca,x_test_pca,y_train,y_test=train_test_split(x_pca,y,test_size=0.2,random_state=30)

model=LogisticRegression(max_iter=1000) # zyada iteration karke Gradient Descent ko converge kr diya.
model.fit(x_train_pca,y_train)
model.score(x_test_pca,y_test)


# In[ ]:




