#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[ ]:


# Here we see there are 50000 training images and 1000 test images


# In[4]:


y_train.shape


# In[5]:


y_train[:5]


# In[6]:


# y_train is a 2D array, for our classification having 1D array is good enough. so we will convert this to now 1D array

y_train = y_train.reshape(-1,)
y_train[:5]


# In[7]:


y_test = y_test.reshape(-1,)


# In[8]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[9]:


# Let's plot some images to see what they are

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[10]:


plot_sample(X_train, y_train, 0)


# In[11]:


plot_sample(X_train, y_train, 1)


# In[ ]:


# Normalize the images to a number from 0 to 1. Image has 3 channels (R,G,B) and
each value in the channel can range from 0 to 255. Hence to normalize in 0-->1 range, we need to divide it by 255


# In[12]:


# Normalizing the training data

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[13]:


# Build simple artificial neural network for image classification

ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[ ]:


# You can see that at the end of 5 epochs, accuracy is at around 49%


# In[14]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[15]:


# Now let us build a convolutional neural network to train our images

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[16]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


cnn.fit(X_train, y_train, epochs=10)


# In[ ]:


# With CNN, at the end 5 epochs, accuracy was at around 70% which is a significant improvement over ANN.

# CNN's are best for image classification and gives superb accuracy.
# Also computation is much less compared to simple ANN as maxpooling reduces 
# the image dimensions while still preserving the features


# In[18]:


cnn.evaluate(X_test,y_test)


# In[19]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[20]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[21]:


y_test[:5]


# In[22]:


plot_sample(X_test, y_test,3)


# In[23]:


classes[y_classes[3]]


# In[ ]:




