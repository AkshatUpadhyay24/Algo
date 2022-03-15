#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implementation of stochastic and batch grandient descent in python.

# We will use very simple home prices data set to implement batch and stochastic gradient descent in python,
# Batch gradient descent uses all training samples in forward pass to calculate cumulitive error and than,
# we adjust weights using derivaties. In stochastic GD, we randomly pick one training sample, perform forward pass, 
# compute the error and immidiately adjust weights. So the key difference here is that to adjust weights batch GD will use,
# all training samples where as stochastic GD will use one randomly picked training sample.


# In[13]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv("homeprice2.csv")
df.sample(5)


# In[15]:


df.columns


# In[16]:


df.drop([ 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5',
       'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
       'Unnamed: 11', 'Unnamed: 12'],axis=1,inplace=True)
df.head()


# In[17]:


# Preprocessing/Scaling: Since our columns are on different sacle it is important to perform scaling on them.

from sklearn import preprocessing

sx = preprocessing.MinMaxScaler()  # 'MinMaxScaler' is used to bring them into (0 to 1) range. 
sy = preprocessing.MinMaxScaler()

scaled_X = sx.fit_transform(df.drop('price',axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0],1))

scaled_X


# In[18]:


scaled_y


# In[8]:


# We should convert target column (i.e. price) into one dimensional array, 
# It has become 2D due to scaling that we did above but now we should change to 1D

scaled_y.reshape(6,)


# In[ ]:


# Gradient descent allows you to find weights (w1,w2,w3) and bias in following linear equation for housing price prediction.

# Price = W1 * area + W2 * bedroom + bias


# In[19]:


# Now is the time to implement mini batch gradient descent.

def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):

    number_of_features = X.shape[1] # 2 features (area,bedroom).
    
    # numpy array with 1 row and columns equal to number of features. In
    
    # our case number_of_features = 2 (area, bedroom)
    
    w = np.ones(shape=(number_of_features)) # it will show values of ( w1 & w2)
    
    b = 0 # bias=0
    
    total_samples = X.shape[0] # number of rows in X
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs): # run epochs 1 by 1.       
        y_predicted = np.dot(w, X.T) + b  #( y_predicted = W1 * area + W2 * bedroom + bias)(T = transpose, use to cnvrt rw into columns & column to row.)
        
        # finding w-gradient and b-gradient here:
        
        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted)) # (1/n)*np.dot(np.transpose(x),(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted) # np.mean(y_true,y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.mean(np.square(y_true-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0: # for optimal value of cost,weights,bias.(it stops at every 10,20,30)
            
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_X,scaled_y.reshape(scaled_y.shape[0],),500) # calling func
w, b, cost


# In[ ]:


# (       w1   ,  w2      ),        b             ,        cost          )


# In[20]:


# Now plot epoch vs cost graph to see how cost reduces as number of epoch increases.

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list,cost_list)


# In[21]:


# Lets do some predictions now.

def predict(area,bedrooms,w,b):
    scaled_X = sx.transform([[area, bedrooms]])[0]
    # here w1 = w[0] , w2 = w[1], w3 = w[2] and bias is b
    # equation for price is w1*area + w2*bedrooms + w3*age + bias
    # scaled_X[0] is area
    # scaled_X[1] is bedrooms
    # scaled_X[2] is age
    scaled_price = w[0] * scaled_X[0] + w[1] * scaled_X[1] + b
    # once we get price prediction we need to to rescal it back to original value
    # also since it returns 2D array, to get single value we need to do value[0][0]
    return sy.inverse_transform([[scaled_price]])[0][0]

predict(2600,4,w,b)


# In[22]:


predict(1000,2,w,b)


# In[23]:


predict(1500,3,w,b)


# In[ ]:


# (2) Stochastic Gradient Descent Implementation

# Stochastic GD will use randomly picked single training sample to calculate error and using this error,
# we backpropage to adjust weights.


# In[24]:


# we will use random libary to pick random training sample for SGD.
import random
random.randint(0,6) # randit gives random number between two numbers specified in the argument


# In[28]:


def SGD(X, y_true, epochs, learning_rate = 0.01):
 
    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features. In 
    # our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):    
        random_index = random.randint(0,total_samples-1) #  use random index from total samples in SGD
        sample_x = X[random_index]
        sample_y = y_true[random_index]
        
        y_predicted = np.dot(w, sample_x.T) + b #( y_predicted = W1 * area + W2 * bedroom + bias)(T = transpose, use to cnvrt rw into columns & column to row.)
    
        w_grad = -(2/total_samples)*(sample_x.T.dot(sample_y-y_predicted))
        b_grad = -(2/total_samples)*(sample_y-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.square(sample_y-y_predicted)  # MSE (Mean Squared Error)
        
        if i%100==0: # at every 100th iteration record the cost and epoch value
            
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = SGD(scaled_X,scaled_y.reshape(scaled_y.shape[0],),10000)
w_sgd, b_sgd, cost_sgd


# In[29]:


# Compare this with weights and bias that we got using gradient descent. They both of quite similar.

w,b


# In[30]:


plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list_sgd,cost_list_sgd)


# In[31]:


predict(2600,4,w_sgd, b_sgd) 


# In[32]:


predict(1000,2,w_sgd, b_sgd)


# In[33]:


predict(1500,3,w_sgd, b_sgd)


# In[ ]:




