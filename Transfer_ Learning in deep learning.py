#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import PIL
import PIL.Image as Image
import os

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])


# In[3]:


IMAGE_SHAPE+(3,)


# In[4]:


gold_fish = Image.open("goldfish.jpg").resize(IMAGE_SHAPE)   # use resize to make pic small.
gold_fish


# In[5]:


gold_fish = np.array(gold_fish)/255.0   # Before classification we need to scale it first.
gold_fish.shape


# In[6]:


gold_fish[np.newaxis, ...]  # these values are in 0 to 1 range.


# In[7]:


result = classifier.predict(gold_fish[np.newaxis, ...])  # newaxis add new dimention in it bcz prediction need multiple images.
result.shape


# In[8]:


predicted_label_index = np.argmax(result)  # argmax will give the index which has maximum value.
predicted_label_index


# In[9]:


# tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

image_labels = []
with open("ImageNetLabels.txt", "r") as f: # open perticular file.
    image_labels = f.read().splitlines()  # split the lines.
image_labels[:5]


# In[10]:


image_labels[predicted_label_index]


# In[11]:


# Load flowers dataset.
# will do classification of flower data sets.

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)

# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it


# In[12]:


data_dir


# In[13]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[14]:


# now we need to convert string path into window path.

list(data_dir.glob('*/*.jpg'))[:5]


# In[15]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[16]:


sunflowers = list(data_dir.glob('sunflowers/*'))
sunflowers[:5]


# In[17]:


PIL.Image.open(str(sunflowers[0]))


# In[18]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[19]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[20]:


# Now will assign class number to each of these flowers randomly, bcz ML not understand text.

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[21]:


flowers_images_dict['sunflowers'][:5]


# In[22]:


str(flowers_images_dict['sunflowers'][0])


# In[23]:


# imread means image read.

img = cv2.imread(str(flowers_images_dict['sunflowers'][1]))


# In[24]:


img.shape


# In[25]:


cv2.resize(img,(224,224)).shape


# In[26]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():  # going to that upper dictionary , going to each rows.
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[27]:


X = np.array(X)
y = np.array(y)


# In[28]:


# Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[29]:


# Preprocessing: scale images

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# In[30]:


# Make prediction using pre-trained model on new flowers dataset


# In[31]:


X[0].shape


# In[32]:


IMAGE_SHAPE+(3,)


# In[33]:


x0_resized = cv2.resize(X[0], IMAGE_SHAPE)
x1_resized = cv2.resize(X[1], IMAGE_SHAPE)
x2_resized = cv2.resize(X[2], IMAGE_SHAPE)


# In[34]:


plt.axis('off')
plt.imshow(X[0])


# In[35]:


plt.axis('off')
plt.imshow(X[1])


# In[36]:


plt.axis('off')
plt.imshow(X[2])


# In[37]:


# it will give predicted array,and argmax will give the maximum argument.

predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized]))
predicted = np.argmax(predicted, axis=1)
predicted


# In[38]:


image_labels[795]


# In[39]:


# Now take pre-trained model and retrain it using flowers images.
# trainable parameter is used to freeze,freeze means do not train and all the layers hv fix weight.

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# In[40]:


# now create a model.

num_of_flowers = 5

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,    # putting ready made model, is already trained.
  tf.keras.layers.Dense(num_of_flowers)  # creating last layer.
])

model.summary()


# In[41]:


model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)


# In[43]:


model.evaluate(X_test_scaled,y_test)


# In[ ]:


#  Through "Transfer Learning" it used only 5 epochs and give 97% accuracy, less consumption of GPU and CPU.

