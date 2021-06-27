# Production script for Podmyne:
# Imports:
import pandas as pd
import json
from flask import Flask, request
# from joblib import Parallel, delayed
from flask_cors import CORS, cross_origin

# globally read data
df = pd.read_pickle('./podcast_data.pkl')

# Init app:
app = Flask(__name__)

# Enable CORS:
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Routes:
# include homepage front URL:
@app.route('/')
def index():
    return app.send_static_file('index.html')

# POST route for recommendations:
@app.route('/app/recommend',methods=['POST'])
def recommend():
    data = request.form.getlist(key='form_data')
    form = data[0].split(',')
    # result = Parallel(n_jobs=-1,backend='multiprocessing')(delayed(get_recommendation)(form)for i in form)
    # print(result)
    result = get_recommendation(form)
    # result = json.dumps(result)
    return result

# Unused route for podcast episodes:
@app.route('/app/episodes',methods=['POST'])
def episodes():
    data = request.form.getlist(key='form_data')
    form = data[0]
    result = episode_finder(form)
    return result

# Function for Cosine Similarity based Recommendation System:
def get_recommendation(user_input):
    # imports
    import json
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    
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
                                                                                'interview','Interviews','interviews']
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), stop_words = stop_words)
    
    # fit and transform on df
    tfdf = tf.fit_transform(df1['all_text'])
    
    # use linear_kernel to create array of similarities
    similarity = linear_kernel(tfdf,tfdf)
    
    # https://www.kaggle.com/switkowski/building-a-podcast-recommendation-engine - adjusted
    x = df1.index[-1]
    similar_idx = similarity[x].argsort(axis = 0)[-21:-1]
    
    # create df of recommendations
    similarity_score = []
    podcast = []
    #description = []
    
    for i in similar_idx:
        similarity_score.append(similarity[x][i])
        podcast.append(df1['title'][i])
        #description.append(df1['description'][i])
        
    recommendations = pd.DataFrame(
    {'similarity_score': similarity_score,
     'title': podcast})
     #'description': description})
    
    # merge data frames
    recommendations = pd.merge(recommendations, df, on='title')
    
    # sort
    recommendations.sort_values(by=['similarity_score'], ascending=False, inplace=True)
    
    #drop unneeded columns
    recommendations.drop(columns=['all_text', 'cat_list'], inplace=True)
    
    # convert df to json
    # reco_json = recommendations.to_json(orient = 'records')
    reco_json = json.dumps(recommendations.to_dict(orient='records'))
    return reco_json

def episode_finder(itunes_id):
    from bs4 import BeautifulSoup as bs
    import requests
    import podsearch as ps
    import pandas as pd
    import json
    
    itunes_id = int(itunes_id)
    urls = []

    # get URLs for most recent 10 episodes
    try:
        for i in range(10):

            pod_url = ps.get(itunes_id).url
            url = pod_url + '/id' + str(itunes_id)
            req = requests.get(url)
            soup = bs(req.content)
            ol = soup.find('ol').find_all('a',href=True)

        for i in ol:
            urls.append(i['href'])
    except:
        urls.append('Not Available')
    
    df = pd.DataFrame(urls,columns=['episode_url'])
    json_urls = json.dumps(df.to_dict(orient='records'))
    return json_urls


# run server: Debugger is active for both dev and production.
if __name__ == '__main__':
    app.run(debug=False)