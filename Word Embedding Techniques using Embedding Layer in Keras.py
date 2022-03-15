#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Word Embedding Techniques using Embedding Layer in Keras


# In[16]:


##tensorflow >2.0

from tensorflow.keras.preprocessing.text import one_hot # use to convert sentence into 'one hot'.


# In[17]:


### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]


# In[18]:


sent


# In[19]:


### Vocabulary size.

voc_size=10000


# In[20]:


# One Hot Representation.

onehot_repr=[one_hot(words,voc_size)for words in sent] # we will get index of each word from dictionary.
print(onehot_repr)


# In[21]:


# Word Embedding Represntation.

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences # 'pad sequence' maintain the size of sentences.
from tensorflow.keras.models import Sequential


# In[22]:


import numpy as np


# In[23]:


sent_length=8 # called padding ( 8 word sentences.), we can give any size it is not fix.

# padding = pre, means as per lengh 8, sentence shd thr so it put words size as per sencence remaining put 0 in front.

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs) # It will show all sentences's index in metric form.


# In[24]:


dim=10 # dimension means features.


# In[25]:


model=Sequential()

model.add(Embedding(voc_size,10,input_length=sent_length))

model.compile('adam','mse') # optimizers.


# In[26]:


model.summary() # it shows embedding layer.


# In[27]:


print(model.predict(embedded_docs)) # To see how words converted into featurized vectors.


# In[28]:


embedded_docs[0] # It shows first sentence in vector form.


# In[29]:


print(model.predict(embedded_docs)[0]) # These are the 10 Dimension vector of each words of first sentence.


# In[ ]:


####  This is Embedding Metrics,through which from 1 sentence, we able to get whole vector Representation by the help of, 
 ##     embedding layers.  ....

