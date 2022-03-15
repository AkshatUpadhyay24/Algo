#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("perrin-freres-monthly-champagne-.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[7]:


df=df.dropna()


# In[8]:


df.isnull().sum()


# In[9]:


df.columns=['Month','Sales']


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df['Month']= pd.to_datetime(df.Month)


# In[14]:


df.info()


# In[15]:


df.set_index('Month',inplace=True)


# In[16]:


df.head()


# In[17]:


df.plot()


# In[18]:


from statsmodels.tsa.stattools import adfuller


# In[19]:


result=adfuller(df.Sales)


# In[20]:


result


# Differencing

# In[22]:


df['Salesdiff']=df['Sales']-df['Sales'].shift(1)


# In[23]:


df.head()


# In[24]:


df['Salesdiff12']=df['Sales']-df['Sales'].shift(12)


# In[25]:


df


# In[26]:


df=df.dropna()


# In[27]:


df.Salesdiff12.plot()


# In[28]:


df.Salesdiff.plot()


# In[29]:


adfuller(df.Salesdiff12)


# In[31]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[32]:


plot_acf(df.Salesdiff12,lags=40)


# In[33]:


plot_pacf(df.Salesdiff12,lags=40)


# In[36]:


plot_acf(df.Sales,lags=40)


# In[35]:


df


# In[38]:


from statsmodels.tsa.arima_model import ARIMA


# In[43]:


model=ARIMA(df.Sales,order=(1,1,1))
model_fit= model.fit()


# In[44]:


model_fit.summary()


# In[46]:


model_fit.predict().plot()


# In[48]:


df.Sales.plot()


# In[50]:


import statsmodels.api as sm 
model=sm.tsa.statespace.SARIMAX(df.Sales,order=(1,1,1),seasonal_order=(1,1,1,12))


# In[51]:


results=model.fit()


# In[54]:


a=results.predict()


# In[55]:


a.plot()


# In[ ]:




