import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from geolocator import *
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def no_stopwords(text):
    tokenwords = word_tokenize(text)
    result = [w for w in tokenwords if not w in stop_words]
    result = []
    for w in tokenwords:
        if w not in stop_words:
            result.append(w)
    return " ".join(result)


def prep_text(df):
    df['sentiment_category'] = df['sentiment'].astype('category')
    df['sentiment_numeric'] = pd.factorize(df['sentiment_category'])[0]
    df['cleaned_tweet'] = df['tweet'].str.replace(
        'https\S+|http\S+|www.\S+|@.\S+|&amp;.\S+|<.*?>', '', case=False)
    df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()
    df['cleaned_tweet'] = df['cleaned_tweet'].str.strip()
    df['normalized_tweet'] = df['cleaned_tweet'].str.translate(
        str.maketrans('', '', string.punctuation))
    df['normalized_tweet'] = df['normalized_tweet'].apply(no_stopwords)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['country'] = df.apply(
        lambda x: get_country(x['long'], x['lat']), axis=1)
    return df
