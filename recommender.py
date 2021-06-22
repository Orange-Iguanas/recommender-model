#!/usr/bin/env python
# coding: utf-8

# # Data Modeling & Recommender

# In[2]:


import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[3]:


# read into model data
df = pd.read_csv('data/model_data.csv')


# In[4]:


# drop unnamed column
df.drop(columns='Unnamed: 0', inplace=True)


# In[5]:


df.head()


# ## Build elements of recommender

# ### Data filtering and prep 

# In[5]:


# create an input box for category
cat_input = input('Enter category: ')


# In[6]:


# create a list of indices of podcasts that fall under the user input category
idx = []
for i in range(df.shape[0]):
    if cat_input in df['cat_list'][i]:
        idx.append(i)


# In[7]:


# create a new df of the podcasts that fall under the user input category
df1 = df.iloc[idx]
df1.reset_index(drop=True, inplace=True)


# In[8]:


df1


# In[10]:


# create an input box for blurb
user_text = input('Describe what you are in the mood to listen to: ')


# In[11]:


# from Patrick Cudo to create a new df of the user input blurb
df2 = pd.DataFrame(columns=['all_text'])
df2['all_text'] = [user_text]
df2


# In[12]:


# append the user input df to the filtered df
df1 = df1.append(df2, ignore_index=True)
df1.tail()


# ### Modeling

# In[13]:


# from Aaron Hume to create add custom words to stop_words list
stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['Podcast','podcast','Podcasts','podcasts']


# In[14]:


# initializing Tfidf and fit transform on df
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
tfdf = tf.fit_transform(df1['all_text'])
tfdf


# In[17]:


# find similarity score between the descriptions
similarity = linear_kernel(tfdf, tfdf)
similarity


# In[19]:


df1.index[-1]


# In[20]:


# https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine

x = df1.index[-1]
similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
for i in similar_idx:
    print(similarity[x][i], '-', df1['title'][i], '-', df1['description'][i], '\n')
print('Original - ' + df1['all_text'][x])


# ## Function for recommender

# In[25]:


# Patrick inspo
def test(df):
    # get category to filter data
    cat_input = input('Enter category: ')
    
    # filter data by category, create new df, reset index - adjusted
    idx = []
    for i in range(df.shape[0]):
        if cat_input in df['cat_list'][i]:
            idx.append(i)
            
    df1 = df.iloc[idx]
    df1.reset_index(drop=True, inplace=True)
    
    # get input string from user
    user_text = input('Describe what you are in the mood to listen to: ')
    
    # add user input to df
    df2 = pd.DataFrame(columns=['all_text'])
    df2['all_text'] = [user_text]

    df1 = df1.append(df2, ignore_index=True)
    
    # instantiate TfidfVectorizer
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['Podcast','podcast','Podcasts','podcasts']
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
    # fit and transform on filtered df
    tfdf = tf.fit_transform(df1['all_text'])
    # use linear_kernel to create array of similarities
    similarity = linear_kernel(tfdf,tfdf)
    # https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine - adjusted
    x = df1.index[-1]
    print (x)
    similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
    for i in similar_idx:
        print(similarity[x][i], '-', df1['title'][i], '-', df1['description'][i], '\n')
    print('Original - ' + df1['all_text'][x])


# In[32]:


test(df)


# In[29]:


df['primary_cat'].unique()


# In[33]:


# True crime is missing as a category, some results are bad (ie. bad parenting)


# ## Script for App

# In[19]:


## create script to take data already filtered by category + user input blurb in json, 
## run model/recommender, 
## return recommendations as a df in json

import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def api_test(json_data):
#     # get category to filter data
#     cat_input = input('Enter category: ')
    
#     # filter data by category, create new df, reset index - adjusted
#     idx = []
#     for i in range(df.shape[0]):
#         if cat_input in df['cat_list'][i]:
#             idx.append(i)
            
#     df1 = df.iloc[idx]
#     df1.reset_index(drop=True, inplace=True)
    
#     # get input string from user
#     user_text = input('Describe what you are in the mood to listen to: ')
    
#     # add user input to df
#     df2 = pd.DataFrame(columns=['all_text'])
#     df2['all_text'] = [user_text]

#     df1 = df1.append(df2, ignore_index=True)

    # convert json to pandas dataframe
    df1 = pd.DataFrame(json_data)
    
    # instantiate TfidfVectorizer
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['Podcast','podcast','Podcasts','podcasts']
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
    
    # fit and transform on all_text column
    tfdf = tf.fit_transform(df1['all_text'])
    
    # use linear_kernel to create array of similarities
    similarity = linear_kernel(tfdf,tfdf)
    
    # https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine - adjusted
    x = df1.index[-1]
    similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
    
    # create df of recommendations
    similarity_score = []
    podcast = []
    description = []
    
    for i in similar_idx:
        similarity_score.append(similarity[x][i])
        podcast.append(df1['title'][i])
        description.append(df1['description'][i])

    recommendations = pd.DataFrame(
    {'similarity_score': similarity_score,
     'podcast': podcast,
     'description': description})
    
    return recommendations


# In[16]:


## test

def test(df):
    # get category to filter data
    cat_input = input('Enter category: ')
    
    # filter data by category, create new df, reset index - adjusted
    idx = []
    for i in range(df.shape[0]):
        if cat_input in df['cat_list'][i]:
            idx.append(i)
            
    df1 = df.iloc[idx]
    df1.reset_index(drop=True, inplace=True)
    
    # get input string from user
    user_text = input('Describe what you are in the mood to listen to: ')
    
    # add user input to df
    df2 = pd.DataFrame(columns=['all_text'])
    df2['all_text'] = [user_text]

    df1 = df1.append(df2, ignore_index=True)
    
    # instantiate TfidfVectorizer
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['Podcast','podcast','Podcasts','podcasts']
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
    # fit and transform on filtered df
    tfdf = tf.fit_transform(df1['all_text'])
    # use linear_kernel to create array of similarities
    similarity = linear_kernel(tfdf,tfdf)
    # https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine - adjusted
    x = df1.index[-1]
    similar_idx = similarity[x].argsort(axis = 0)[-6:-1]

    similarity_score = []
    podcast = []
    description = []
    
    for i in similar_idx:
        similarity_score.append(similarity[x][i])
        podcast.append(df1['title'][i])
        description.append(df1['description'][i])

    recommendations = pd.DataFrame(
    {'similarity_score': similarity_score,
     'podcast': podcast,
     'description': description})
    
    return recommendations


# In[17]:


test(df)


# In[ ]:




