# Function for retrieving URLs for itunes episode pages by API calls from Itunes API
# podsearch is required 3rd party installation

def episode_finder(df):
    from bs4 import BeautifulSoup as bs
    import requests
    import podsearch as ps
    # df = pd.read_pickle('./podcast_data.pkl')
    for i in range(4):
        try:
            pod_url = ps.get(df['itunes_id'][i]).url
            pod_id = str(df['itunes_id'][i])
            url = pod_url + '/id' + pod_id
            req = requests.get(url)
            soup = bs(req.content)
            ol = soup.find('ol').find_all('a',href=True)
#             convert to json array output instead of printing
            for i in ol:
                print(i['href'])
        except:
            print('Not Available')