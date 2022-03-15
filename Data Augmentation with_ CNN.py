#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)


# In[3]:


data_dir


# In[4]:


# convert data directory into pathlib first, pathlib is a window path object.

import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[5]:


# glob will give all the images which has extension Jpg.

list(data_dir.glob('*/*.jpg'))[:5]


# In[6]:


# total jpg images.
   
   image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[7]:


# This will give images of sunflower.

sunflowers = list(data_dir.glob('sunflowers/*'))
 sunflowers[:5]


# In[8]:


PIL.Image.open(str(sunflowers[3]))


# In[9]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[10]:


# Now dictionary have all the image path.

flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[11]:


# Assigning class number to each of these flowers Randomly.

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[12]:


flowers_images_dict['sunflowers'][:5]


# In[13]:


# opencv is not accepting window path as an argument,
# so, by using 'str' it will give actual string path and opencv axpect that string path.

str(flowers_images_dict['sunflowers'][0])


# In[14]:


# opencv has this module 'imread' and in return it will give numpy array.

img = cv2.imread(str(flowers_images_dict['sunflowers'][0]))


# In[15]:


# opencv's imread module 3d array(x,y,rgb)

img.shape


# In[16]:


# opencv has a function resize, because in folder these are many images r of diffrent size.
# so we have to give fix size first.

cv2.resize(img,(180,180)).shape


# In[17]:


# Now, run for loop to prepair X and Y.

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images: # go through all the images.
        img = cv2.imread(str(image))  # read images with string path.
        resized_img = cv2.resize(img,(180,180))  # resize again bcz ML expect all the images should b in same dimensions.
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name]) # y need flower name.


# In[18]:


# convert list into numpy array first.

X = np.array(X)
y = np.array(y)


# In[19]:


# Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[20]:


# Preprocessing: scale images

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# In[21]:


# Build convolutional neural network and train it .
# here we specify our layers 1 by 1.
# filter values r not fix.

num_classes = 5  # bcz we have 5 category of flowers

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'), # (16- filters, size of filters-3), layer 1
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'), # (32- filters, size of filters-3), layer 2
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'), # (64- filters, size of filters-3), layer 3
  layers.MaxPooling2D(),
    
    # before dense network we have to flatten our values bcz dense network,
    # accept single dimention array by specify the flatten layer.
    
  layers.Flatten(),
  layers.Dense(128, activation='relu'), # 128- neurons
  layers.Dense(num_classes) # here we did not mentn activation bcz it will use linear activation.
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=12)              


# In[22]:


model.evaluate(X_test_scaled,y_test)


# In[23]:


predictions = model.predict(X_test_scaled)
predictions


# In[24]:


# tf has softmax function which can convert numpy array into set of probability between (1 and 0)

score = tf.nn.softmax(predictions[0])


# In[25]:


# argmax function give the index of element which is maximum.

np.argmax(score)


# In[ ]:


# it shows '4' means 'tulips' flower as per the labels divides to each flowers.


# In[26]:


y_test[0]


# In[51]:


# Improve Test Accuracy Using Data Augmentation

data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),
     layers.RandomZoom(0.2),                                
     layers.RandomRotation(0.1), ]) # Rotation means more rotate


# In[52]:


# Original Image

plt.axis('off')
plt.imshow(X[0])


# In[53]:


# Newly generated training sample using 'data augmentation' like - contrast, rotate,zoom in 

plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


# In[54]:


# Train the model using data augmentation and a drop out layer

num_classes = 5

model = Sequential([
  data_augmentation, # data augmentation as a first layer.
    
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),  # it will 20% of neuron at random in each pass to give better generalization.
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=15)    


# In[55]:


model.evaluate(X_test_scaled,y_test)


# In[ ]:


# test set accuracy is 91% now.

