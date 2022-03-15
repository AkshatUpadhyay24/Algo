#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('insurance data.csv')
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[4]:


df1=df.drop(['insurance'],axis=1)
df1.head()


# In[5]:


df.shape


# In[7]:


df.insurance.head()


# In[8]:


x=df1
y=df.insurance


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('age')
plt.ylabel('insurance')
plt.scatter(df.age,df.insurance,color='red',marker='+')


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


model=LogisticRegression()


# In[26]:


model.fit(x_train,y_train)


# In[17]:


df.head()


# In[28]:


model.predict(x_test)


# In[31]:


model.predict([[46]])


# In[32]:


model.score(x_test,y_test)


# In[ ]:




