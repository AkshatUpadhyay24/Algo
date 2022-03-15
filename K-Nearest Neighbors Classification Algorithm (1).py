#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('IRIS.csv')
df.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder
le_species=LabelEncoder()
df['species_n']=le_species.fit_transform(df['species'])
df.head()


# In[4]:


# It will show k 'species_n' column me category 1,2,3 konse index se shuru ho ri hai.

df[df.species_n==1].head()


# In[5]:


# Now, for better visualize we divide data set in 3 diffrent sets.

df1=df[:50]
df2=df[50:100]
df3=df[100:]


# In[6]:


df['species'].value_counts()


# In[12]:


# Now will create scatter graph "Sepal lenght" vs "Sepal Width ".

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df1.sepal_length,df1.sepal_width,color='red')
plt.scatter(df2.sepal_length,df2.sepal_width,color='blue')
plt.scatter(df3.sepal_length,df3.sepal_width,color='green')


# In[14]:


df4=df.drop(['species'],axis=1)
df4.head()


# In[15]:


df5=df4.drop(['species_n'],axis=1)
df5.head()


# In[16]:


x=df5
y=df4.species_n


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[40]:


# Now, will create KNN(K Neighrest Neighbour Classifier)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # n_neighbors, means K value.


# In[41]:


knn.fit(x_train,y_train)


# In[42]:


knn.score(x_test,y_test)


# In[43]:


knn.predict(x_test)


# In[44]:


y_pred=knn.predict(x_test)


# In[46]:


# now will create confusion Matrix.

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[47]:


# Now, visualize CM

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('True')


# In[48]:


# To find Classification Report.

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




