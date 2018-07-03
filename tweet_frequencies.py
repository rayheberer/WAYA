import tweepy
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from flask import Flask, url_for, render_template
app = Flask(__name__)

# twitter authentication
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def process(text, lower=False):
    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9@# ]', '', text)
    # remove twitter shortened urls
    text = re.sub(r'https.*', '', text)
    
    if lower:
        return text.lower()
    else:
        return text

def get_tweets(handle, count=20, **kwargs):
    tweets = api.user_timeline(handle, tweet_mode='extended', count=count, **kwargs)    
    return map(lambda x: process(x.full_text), tweets)

def remove_stopwords(tweets):
    tokenizer = TweetTokenizer()
    tokens = map(tokenizer.tokenize, tweets)
    stops = set(stopwords.words('english'))
    no_stops = map(lambda sent: [x for x in sent if x not in stops and x != 'RT'], 
                   tokens)
    
    return list(map(' '.join, no_stops))
    
def get_top_entities(term_counts, vectorizer, what='@', n=5):
    total_counts = np.squeeze(np.array(term_counts).sum(axis=0))
    entities = []
    
    for term, idx in vectorizer.vocabulary_.items():
        if term.startswith(what):
            entities.append((term, total_counts[idx]))
    
    entities.sort(key=lambda x: x[1], reverse=True)

    entities_dict = {}
    for entity in entities[:n]:
        entities_dict[entity[0]] = entity[1]

    return entities_dict

def remove_twitter_terms(text):
    tokens = text.split()
    tokens = [w for w in tokens if (not w.startswith('#') 
                                    and not w.startswith('@'))]
    
    return ' '.join(tokens)

def make_wordcloud_corpus(tweets):
    corpus = ' '.join(map(remove_twitter_terms, tweets))
    return corpus

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/mentions/<handle>')
def mentions(handle):
    tweets = get_tweets(handle, 100)
    no_stops = remove_stopwords(tweets)

    vectorizer = CountVectorizer(token_pattern=r'(?u)@?#?\b\w\w+\b', lowercase=False)
    vectorizer.fit(no_stops)
    data = vectorizer.transform(no_stops)
    data = data.todense()

    mentions = get_top_entities(data, vectorizer, n=10)
    return str(mentions)

@app.route('/hashtags/<handle>')
def hashtags(handle):
    tweets = get_tweets(handle, 100)
    no_stops = remove_stopwords(tweets)

    vectorizer = CountVectorizer(token_pattern=r'(?u)@?#?\b\w\w+\b', lowercase=False)
    vectorizer.fit(no_stops)
    data = vectorizer.transform(no_stops)
    data = data.todense()

    tags = get_top_entities(data, vectorizer, '#', n=10)
    return str(tags)

@app.route('/wordcloud/<handle>')
def wordcloud(handle):
    tweets = get_tweets(handle, 100)
    no_stops = remove_stopwords(tweets)

    corpus = make_wordcloud_corpus(no_stops)
    wordcloud = WordCloud(background_color='white',
                          max_words=100,
                          max_font_size=40, 
                          random_state=42).generate(corpus)

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    fig.savefig("static/cloud.png", pad_inches=0)
    return "Wordcloud saved at /static/cloud.png"