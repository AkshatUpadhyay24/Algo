#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jo bhi Kth (n-number) each Kth ka Average score hoga wahi called " K_Fold cross Validation" 
# K 10, K 5, K 15, bhi mention kar sakte h.


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv('IRIS.csv')
df.head()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le_species=LabelEncoder()
df['species_n']=le_species.fit_transform(df['species'])
df.head()


# In[5]:


df1=df.drop(['species'],axis=1)
df1.head()


# In[6]:


df2=df1.drop(['species_n'],axis=1)
df2.head()


# In[7]:


x=df2
y=df1.species_n


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[13]:


# model performance by LR(Logistic Regression)

lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[14]:


lr.score(x_test,y_test)


# In[15]:


# now, model performance by SVM

svm=SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)


# In[16]:


# Now, model performance by Random Forest.

rf=RandomForestClassifier(n_estimators=40)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)


# In[17]:


# now import K-Fold .

from sklearn.model_selection import KFold
kf=KFold(n_splits=3) # means create 3 folds
kf


# In[18]:


from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(),x,y)


# In[19]:


cross_val_score(SVC(),x,y)


# In[20]:


cross_val_score(RandomForestClassifier(),x,y)


# In[23]:


# now will average the score.

scores1=cross_val_score(RandomForestClassifier(n_estimators=5),x,y,cv=10)
np.average(scores1)


# In[24]:


# now for 20 trees of RF.

# cv=10, means 10 folds

scores2=cross_val_score(RandomForestClassifier(n_estimators=20),x,y,cv=10)
np.average(scores2)


# In[25]:


# now for 30 trees of RF.

scores3=cross_val_score(RandomForestClassifier(n_estimators=30),x,y,cv=10)
np.average(scores3)


# In[ ]:


# so, we can say in RF we need only 5 Trees for good accuracy. 

