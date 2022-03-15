#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import cv2


# In[2]:


DATADIR="C:\\Users\\akshansh\\Desktop\\Deep learning codes\\animal images\\animals"
CATEGORIES=['cats','dogs','panda']


# In[3]:


for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break


# In[4]:


print(img_array)


# In[5]:


print(img_array.shape)


# In[6]:


img_size=20

new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()


# In[7]:


print(new_array.shape)


# In[8]:


training_data=[]

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category) # path to cats or dogs and panda dir.
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            
create_training_data()


# In[9]:


print(len(training_data))


# In[10]:


import random

random.shuffle(training_data)


# In[11]:


for sample in training_data[:10]:
    print(sample[1])
    
# 0=cat,1=dog,2=panda.


# In[12]:


x=[]
y=[]


# In[13]:


for features, label in training_data:
    x.append(features)
    y.append(label)
    
x=np.array(x).reshape(-1,img_size,img_size,1)


# In[14]:


import pickle

pickle_out=open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[15]:


pickle_in= open("x.pickle","rb")
x=pickle.load(pickle_in)


# In[16]:


x[1]


# In[17]:


len(x)


# In[18]:


len(y)


# In[19]:


plt.imshow(x[2])


# In[24]:


x=np.asarray(pickle.load(open('x.pickle','rb')))

y=np.asarray(pickle.load(open('y.pickle','rb')))


# In[25]:


# now we normalize the value

x=x/255.0


# In[26]:


x[1]


# In[29]:


model=Sequential()

# layer 1

model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3

model.add(Flatten())
model.add(Dense(64))

# Output layer

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=10, validation_split=0.1)


# In[ ]:




