#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


df=pd.read_csv('IRIS.csv')
df.head()


# In[30]:


df.shape


# In[31]:


df['species'].unique()


# In[32]:


from sklearn.preprocessing import LabelEncoder
le_species=LabelEncoder()
df['species_n']=le_species.fit_transform(df['species'])
df


# In[33]:


df1=df.drop(['species'],axis=1)
df1.head(2)


# In[42]:


df2=df1.drop(['species_n'],axis=1)
df2.head()


# In[43]:


df1.species_n.head()


# In[44]:


x=df2
y=df1.species_n


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[50]:


from sklearn.svm import SVC
model=SVC()


# In[51]:


model.fit(x_train,y_train)


# In[52]:


model.score(x_test,y_test)


# In[58]:


df[100:150].head()


# In[59]:


model.predict([[7.1,3.0,5.9,2.1]])


# In[ ]:


# we can also apply parameters in SVM of improve Accuracy.

# Regularization Parameter(c)
# Gamma Parameter
# Linear_kernel parameter

# -->> from sklearn.model_selection import train_test_split
# -->> x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# -->> model= svm.SVC(kernel='rbf',c=30,gamma='auto')

# -->> model.fit(x_train,y_train)

# -->> model.score(x_test,y_test)

