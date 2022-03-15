#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Neural network for handwritten digits classification using TF.

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[3]:


len(x_train)


# In[4]:


len(x_test)


# In[5]:


x_train[0].shape # sample has 28 by 28 pixel image.


# In[6]:


x_train[0] # between 0 to 255 value in 2D array.

# 0- black
# 255- white


# In[7]:


# plotting first training image.

plt.matshow(x_train[0])


# In[8]:


# 2nd image

plt.matshow(x_train[1])


# In[9]:


# 'y-train' will show us the actual digits

y_train[0]


# In[10]:


y_train[1]


# In[11]:


# first 5 ac tual digits

y_train[:5]


# In[15]:


x_train.shape


# In[16]:


# now we need to scale it, so that values will be in between 0 to 1, it also helps to increase accuracy .

x_train=x_train/255
x_test=x_test/255


# In[44]:


x_test.shape


# In[17]:


x_train.shape


# In[40]:


# now will reshape it, because we need to convert 2D array into 1D.

x_train_f=x_train.reshape(len(x_train),28*28)
x_test_f=x_test.reshape(len(x_test),28*28)


# In[42]:


x_train_f.shape


# In[43]:


x_test_f.shape


# In[45]:


x_train_f[0] # 2D converted into 1D.


# In[ ]:


# input layer has - 784 elements
# output layer has - 10 elements


# In[21]:


# now will define Neural Network.
# Sequential means stack of layers in NN.
# Dense means neurons in one layer connected to the neuron in secound layer.

model=keras.Sequential([keras.layers.
                        Dense(10,input_shape=(784,),activation='sigmoid')]) # output layer,then input layer n activation.


# In[22]:


# "optimizer" allow us to train efficiently when the backward prapogation and training is going on,
# -->> it allow us to reach global optima

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[23]:


model.fit(x_train_f,y_train,epochs=5)


# In[24]:


# After fit we need to evaluate the model.

model.evaluate(x_test_f,y_test)


# In[48]:


# Now going to predict first image.

plt.matshow(x_test[0])


# In[26]:


y_pred=model.predict(x_test_f)


# In[27]:


y_pred[0] # 10 scores of first image are printed in this array.


# In[49]:


# Now, will see the predicted value:

np.argmax(y_pred[0]) # "argmax" will find the maximum value and print the index of that value(y_pred[0]).


# In[29]:


y_test[:5]


# In[30]:


# y_test are integer value & y_pred are whole value.
# -->> so, we need to convert y_pred into 'concrete class label.'

y_pred_labels=[np.argmax(i) for i in y_pred] # for each "i" we need to find np.argmax.
y_pred_labels[:5]


# In[ ]:


# So, first five predicted matching with the Truth Data.


# In[31]:


# now will check for confusion matrix

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)
cm


# In[33]:


# now, will show confusion matrix into colorfull visualization:

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')

# All the numbers are error except of below diagnal.


# In[34]:


# now will put "hidden layer" in neural network
# "Hidden layer " helps to improve the accuracy of model.

model=keras.Sequential([keras.layers.Dense(100,input_shape=(784,),activation='relu'),#-->> "Hidden Layer"
                        
                       keras.layers.Dense(10,activation='sigmoid')
                        
                       ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[36]:


model.fit(x_train_f,y_train,epochs=8)


# In[37]:


model.evaluate(x_test_f,y_test)


# In[38]:


# Now, again check cm by visualize.

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:


# So, we found that error become less in black colors.


# In[50]:


p=np.array([1000,400,1200])
u=np.array([[30,40,50],[5,10,15],[2,5,7]])


# In[51]:


np.dot(p,u)


# In[52]:


y_pred=np.array([1,1,0,0,1])
y_true=np.array([0.30,0.7,1,0,0.5])


# In[53]:


np.mean(np.abs(y_pred-y_true))


# In[54]:


np.sum(np.abs(y_pred-y_true))


# In[ ]:




