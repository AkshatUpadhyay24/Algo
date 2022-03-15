#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[13]:


df=pd.read_csv('IRIS.csv')
df.head()


# In[14]:


from sklearn.preprocessing import LabelEncoder
le_species=LabelEncoder()
df['species_n']=le_species.fit_transform(df['species'])
df.head()


# In[20]:


df1=df.drop(['species'],axis=1)
df1.head()


# In[22]:


df2=df1.drop(['species_n'],axis=1)
df2.head()


# In[23]:


x=df2
y=df1.species_n


# In[83]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[84]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)


# In[85]:


model.fit(x_train,y_train)


# In[86]:


model.score(x_test,y_test)


# In[91]:


df[100:150].head()


# In[92]:


model.predict([[7.1,3.0,5.9,2.1]])


# In[94]:


model.predict(x_test)


# In[95]:


y_pred=model.predict(x_test)
y_pred


# In[96]:


# create confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[97]:


# now plot confusion mateix.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




