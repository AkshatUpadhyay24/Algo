#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries 

# In[1]:


import pandas as pd
import numpy as np


# ### Import the movies dataset

# In[2]:


movie_df = pd.read_csv(r"E:\DataScience-data\movies.csv")


# In[3]:


rating_df = pd.read_csv(r"E:\DataScience-data\ratings.csv")


# ### Checking the tables 

# In[4]:


movie_df.head(10)


# In[5]:


rating_df.head(10)


# ### Now combine the two tables and drop things we dont have to use

# In[6]:


combine_movie_rating = pd.merge(rating_df, movie_df, on='movieId')
combine_movie_rating.head(10)


# In[7]:


columns = ['timestamp', 'genres']
combine_movie_rating = combine_movie_rating.drop(columns, axis=1)
combine_movie_rating.head(10)


# In[8]:


combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])

movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head(10)


# In[9]:


rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
rating_with_totalRatingCount.head(10)


# ### Now drop the duplicate data

# In[10]:


user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])
user_rating.head(10)


# ## Matrix Factorization

# ### Now create a matrix and fill 0 values  

# In[11]:


movie_user_rating_pivot = user_rating.pivot(index = 'userId', columns = 'title', values = 'rating').fillna(0)
movie_user_rating_pivot.head(10)


# In[12]:


X = movie_user_rating_pivot.values.T
X.shape


# ### Now lets fit the model

# In[13]:


import sklearn
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape


# In[14]:


import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape


# ### Now lets check the results

# In[27]:


movie_title = movie_user_rating_pivot.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Guardians of the Galaxy (2014)")


# In[28]:


corr_coffey_hands  = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])

