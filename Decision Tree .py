#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('salarydata.csv')
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[5]:


for i in df.columns:
    
    print(df[i].value_counts())


# In[6]:


from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
df['company_n']=le_company.fit_transform(df['company'])
df['job_n']=le_company.fit_transform(df['job'])
df['degreee_n']=le_company.fit_transform(df['degree'])
df


# In[7]:


df1=df.drop(['company','job','degree'],axis=1)
df1.head()


# In[9]:


df2=df1.drop(['salary'],axis=1)
df2.head()


# In[10]:


df1.salary.head()


# In[11]:


x=df2
y=df1.salary


# In[12]:


from sklearn import tree
model=tree.DecisionTreeClassifier()


# In[13]:


model.fit(x,y)


# In[14]:


model.score(x,y)


# In[16]:


df.head()


# In[19]:


model.predict([[2,0,0]])


# In[ ]:




