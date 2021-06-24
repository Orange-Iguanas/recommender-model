#!/usr/bin/env python
# coding: utf-8

# # Final Model for App

# In[25]:


# function to process data and output recommendations
# user input is a list of two strings: category, interests + blurb

def get_recommendation(user_input):
    # imports
    import numpy as np
    import pandas as pd 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    
    # get data
    df = pd.read_pickle('podcast_data.pkl')
    
    # get category to filter data
    cat_input = user_input[0]
    
    # filter data by category, create new df, reset index - adjusted
    idx = []
    for i in range(df.shape[0]):
        if cat_input in df['cat_list'][i]:
            idx.append(i)
            
    df1 = df.iloc[idx]
    df1.reset_index(drop=True, inplace=True)
    
    # get input string from user
    user_text = user_input[1]
    
    # add user input to df
    df2 = pd.DataFrame(columns=['all_text'])
    df2['all_text'] = [user_text]

    df1 = df1.append(df2, ignore_index=True)
    
    # instantiate TfidfVectorizer
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['Podcast','podcast','Podcasts','podcasts',
                                                                                'Live','live','Radio','radio','Show','show',
                                                                                'interview','Interviews']
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
    
    # fit and transform on df
    tfdf = tf.fit_transform(df1['all_text'])
    
    # use linear_kernel to create array of similarities
    similarity = linear_kernel(tfdf,tfdf)
    
    # https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine - adjusted
    x = df1.index[-1]
    similar_idx = similarity[x].argsort(axis = 0)[-9:-1]
    
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
    
    # convert df to json
    reco_json = recommendations.to_json(orient = 'records')
    
    return reco_json


# In[ ]:




