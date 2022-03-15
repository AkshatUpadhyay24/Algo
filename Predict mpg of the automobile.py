#!/usr/bin/env python
# coding: utf-8

# ### Importing the LIbraries

# In[1]:


import pandas as pd #data processing, I/O operation
import numpy as np #linear algebra
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# ### loading the dataset

# In[2]:


data = pd.read_csv("/Users/sachingarg/Desktop/auto-mpg.csv")
data.head(10)


# In[3]:


data.tail(10)


# In[4]:


data.drop(['car name'], axis=1, inplace=True)
data.head()


# In[5]:


# Summary of the Dataset
data.describe()


# ### Data Preprocessing

# In[6]:


data.isnull().sum()


# In[7]:


data['horsepower'].unique()


# In[8]:


data = data[data.horsepower != '?']


# In[9]:


'?' in data


# In[10]:


data.shape


# ### Correlation matrix

# In[11]:


data.corr()['mpg'].sort_values()


# In[12]:


#Plotting the heatmap of the correlation

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, center=0, cmap='rainbow')
plt.show()


# ### Univariate Analysis

# In[14]:


sns.countplot(data.cylinders, data=data, palette='rainbow')
plt.show()


# In[15]:


sns.countplot(data['model year'], palette='rainbow')
plt.show()


# In[17]:


sns.countplot(data.origin, palette='rainbow')
plt.show()


# ### Multi-variate Analysis

# In[18]:


sns.boxplot(y='mpg', x='cylinders', data=data, palette='rainbow')
plt.show()


# In[19]:


sns.boxplot(y='mpg', x='model year', data=data, palette='rainbow')
plt.show()


# In[20]:


# Modelling my dataset

X = data.iloc[:,1:].values
Y = data.iloc[:,0].values


# ### Train and test data split

# In[21]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)


# ### Build the model

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regression = LinearRegression()
regression.fit(x_train,y_train)


# In[24]:


y_pred = regression.predict(x_test)


# In[25]:


print(regression.score(x_test, y_test))


# ### polynomial regression

# In[26]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(X_poly,Y,test_size=0.3, random_state=0)

lin_regression = LinearRegression()
lin_regression.fit(x_train,y_train)

print(lin_regression.score(x_test, y_test))


# ### Conclusion

# Accuracy score improves in the case of polynomial regression compared to the linear regression because it fits data much better. In this project, what we learned:
# 1. Loading the dataset
# 2. Univariate analysis
# 3. multivariate analysis
# 4. Linear regression
# 5. Polynomial Regression

# In[ ]:




