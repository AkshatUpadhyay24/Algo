#!/usr/bin/env python
# coding: utf-8

# In[1]:


# naive_bayes_2_email_spam_filter.

import pandas as pd


# In[2]:


df = pd.read_csv('Spam.csv',encoding='unicode_escape')
df.head()


# In[3]:


df.groupby('Category').describe() # groupby 'category'


# In[4]:


df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0) # convert both columns into integer through 'apply'func.
df.head()                                                        # 'Lambda' function check value from each individual.


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.25)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer # use to convert categorical values into numbers.
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]


# In[7]:


from sklearn.naive_bayes import MultinomialNB # one of the type of naive bayes.
model = MultinomialNB()
model.fit(X_train_count,y_train) # 'X_train_count' is the text converted mail into numbers metric.


# In[8]:


# so, the model is trained now.


# In[9]:


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[10]:


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)


# In[11]:


# Sklearn Pipeline use for same code as above using simple API.

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()), # convert text into vector.
    ('nb', MultinomialNB())
])


# In[12]:


# previous we used 'X_train_count' & then train the model,but here we can directly feed text into model.

clf.fit(X_train, y_train)


# In[13]:


clf.score(X_test,y_test)


# In[14]:


clf.predict(emails)


# In[ ]:




