#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('diabetes.csv',skiprows=1)
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df=df.interpolate()


# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# In[9]:


df1=df.drop(['outcome'],axis=1)
df1.head()


# In[11]:


x=df1
y=df.outcome


# In[13]:


# Now, will do 'Scaling'

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled[:3] # first 3 element


# In[14]:


# Now, will do train,test,split by dividing data sets.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[19]:


# Now, we train the model by using Decision Tree & run cross validation on it.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
scores=cross_val_score(DecisionTreeClassifier(),x,y,cv=5)
scores


# In[20]:


scores.mean()


# In[23]:


# Now, will Apply "Bagging Classifier"

from sklearn.ensemble import BaggingClassifier
bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                          n_estimators=100,
                          max_samples=0.8,
                          oob_score=True, 3 # oob--> out of bag sample
                           random_state=0
                          )


# In[24]:


bag_model.fit(x_train,y_train)
bag_model.oob_score_


# In[25]:


# Now, for actual score.

bag_model.score(x_test,y_test)


# In[37]:


# Now,will create RandomForestClassifier and run Crossvalidation score on it.

from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(n_estimators=50),x,y,
        
                      cv=5)
scores


# In[38]:


scores.mean()


# In[ ]:




