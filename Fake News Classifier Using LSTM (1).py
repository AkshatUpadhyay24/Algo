#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fake News Classifier Using LSTM.


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('Lstm_fakenews.csv')
df.head()


# In[4]:


###Drop Nan Values

df=df.dropna()


# In[5]:


## Get the Independent Features

X=df.drop('label',axis=1)


# In[6]:


## Get the Dependent features

y=df['label']


# In[7]:


X.shape


# In[8]:


y.shape


# In[9]:


import tensorflow as tf


# In[10]:


tf.__version__


# In[11]:


# Neccessary things to Apply LSTM. 

from tensorflow.keras.layers import Embedding # Embedding Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences # 'pad_sequence' use for input length or input size,
# to make sentences equal in length.

from tensorflow.keras.models import Sequential # bcz at last will create sequential model.
from tensorflow.keras.preprocessing.text import one_hot #'one hot' convrt sentnces into one hot encoder in givn vocabulary size.
from tensorflow.keras.layers import LSTM # lstm layer
from tensorflow.keras.layers import Dense # it decide probability if greater thn >.5 is fake, if less than < .5 not fake.


# In[12]:


### Vocabulary size

voc_size=5000


# In[13]:


# Onehot Representation for 'title' column.


# In[14]:


messages=X.copy()


# In[15]:


messages['title'][1]


# In[16]:


# we do reset index after droping na values,so tht it will not drop indexes.

messages.reset_index(inplace=True)


# In[17]:


import nltk
import re
from nltk.corpus import stopwords


# In[18]:


nltk.download('stopwords') # remove unneccesary words by using 'stopwords'.


# In[19]:


### Dataset Preprocessing to clean the data.

from nltk.stem.porter import PorterStemmer # 'porterstemer' use for stemming part.
ps = PorterStemmer()
corpus = [] # creating a list as 'corpus'.

for i in range(0, len(messages)): # for loop will run through all the indexes.
    print(i)
    # re - regular expression.
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i]) # apart from atoz Ato Z, sustituting every thing with a blank,
    # -->> just skipping these character with and all the remaining character replaced with blank space.
    # 'title' is the column name.
    
    review = review.lower() # will do lowering of words of sentences.
    review = review.split() # 'split' to remove stopwords.
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # it means if that word is not present,
    # -->> in stopwords then olny will do stemming.
    
    review = ' '.join(review) # combining list of words.
    
    corpus.append(review)


# In[20]:


corpus


# In[21]:


# In 'one hot representation' we need the index based on vocabulary type.
# 1st parameter - words
# 2nd parameter - vocabulary size.

onehot_repr=[one_hot(words,voc_size)for words in corpus]

onehot_repr # all the words converted into specific index.


# In[22]:


#  now will do 'Embedding Representation'.

# before 'Embedding Representation' we need to make these sentences of fix length by using 'pad sequences'.
# parameters of 'pad sequences'- one hot, padding = pre(for fix size), maxlen = sentence length.


sent_length=20 # we can set sencence length.

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs)

# pre padding will add like 0 0 0 in front.


# In[23]:


embedded_docs[0] # it shows first sentence, 10 zeros(bcz of 20 sentence length) with 10 index numbers of each words.

# 'X' is now present in embedded docs, X is independent feature.


# In[24]:


## Creating model

embedding_vector_features=40 # need to mention vector features( can take 100 also.)
model=Sequential() # sequential layer.
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # perameters of Embedding layer.

model.add(LSTM(100)) # then will pass through lSTM layer( 1 lstm= 100 neurons)

model.add(Dense(1,activation='sigmoid')) # for classification problms we use dense for 1 final output,
 # -->> 'sigmoid' will tell us the probability whether it is from class 1 or class 2.

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

# 'summary' shows embedding layer,lstm layer,dense layer.


# In[25]:


len(embedded_docs),y.shape


# In[26]:


# now will convert embedded docs in an array and storing in X_final,similarly taking 'y' and storing in y_final.

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[27]:


X_final.shape,y_final.shape


# In[28]:


# Test Train Split( to test model)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# In[29]:


# Model Training


# In[30]:


### Finally Training

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64) # select any batch size.


# In[31]:


# Now, Adding Dropout


# In[107]:


from tensorflow.keras.layers import Dropout

## Creating model

embedding_vector_features=42 # we can change it

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model.add(Dropout(0.4)) # Hyper perameter.

model.add(LSTM(100))

model.add(Dropout(0.4)) # Hyper perameter.

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[108]:


# Performance Metrics And Accuracy.

y_pred=(model.predict(X_test)>0.5).astype("int32") #  use this code for 2.8 version of TF.

# y_pred=model.predict_classes(X_test) -->> use this code for 2.1 version of TF


# In[109]:


from sklearn.metrics import confusion_matrix


# In[110]:


confusion_matrix(y_test,y_pred)


# In[111]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




