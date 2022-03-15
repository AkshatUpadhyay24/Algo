#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[5]:


ipl=pd.read_csv("C:\\Users\\akshansh\\Desktop\\python programs\\matches.csv")
                


# In[6]:


ipl.head()


# In[7]:


ipl.shape


# In[8]:


ipl['player_of_match'].value_counts()


# In[9]:


ipl['player_of_match'].value_counts()[0:5]


# In[26]:


# Player of the match

plt.figure(figsize=(8,5))
plt.bar(list(ipl['player_of_match'].value_counts()[0:5].keys()),list(ipl['player_of_match'].value_counts()[0:5]),color="g")
plt.show()


# In[29]:


# Team won most

plt.figure(figsize=(6,6))
plt.bar(list(ipl['winner'].value_counts()[0:3].keys()),list(ipl['winner'].value_counts()[0:3]),color=["blue","yellow","orange"])
plt.show()


# In[31]:


# Team won by highest margin of runs.

plt.figure(figsize=(5,7))
plt.hist(ipl['win_by_runs'])
plt.title('Distribution of runs')
plt.xlabel('Runs')
plt.show()


# In[ ]:




