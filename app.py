from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import nltk

from data_manipulations import *

nltk.download('stopwords')
nltk.download('punkt')

app = Dash(__name__)

df = pd.read_csv("tweets_annotated.1650575029.formulaone.csv")
df = prep_text(df)

# Line - Tweets Over Time Buckets
tweet_df_buckets = df.groupby(pd.Grouper(
    key='created_at', freq='30Min', convention='start')).size()
df_t = pd.DataFrame(list(tweet_df_buckets.items()),
                    columns=['created_at', 'count'])

fig_buckets = px.line(
    df_t,
    x="created_at",
    y="count",
    title="Tweet count in 30 minute buckets"
)

# Lines - Tweet Sentiment Counts in 30 minute Time Buckets
tweet_sentiment_time_buckets = df.groupby([
        'sentiment',
        pd.Grouper(
        key='created_at', freq='30Min', convention='start')
    ]).size()
df_t_s = tweet_sentiment_time_buckets.reset_index(name='counts')

fig_tweet_sentiment_time_buckets = px.line(
    df_t_s,
    line_group="sentiment",
    color="sentiment",
    x="created_at",
    y="counts",
    title="Tweet Sentiment Counts in 30 minute Time Buckets"
)

# Pie - Sentiment
df_sent_counts = df['sentiment'].value_counts().reset_index(name='counts')
fig_pie_overall_sent = px.pie(
    df_sent_counts,
    values='counts',
    names='index',
    title='Sentiment Pie Chart'
)

# Geo - Tweet Counts
df_country_counts = df['country'].value_counts().reset_index(name='counts')
# print(df_country_counts)
fig_geo_tweet_counts = px.scatter_geo(
    df_country_counts,
    locations="index",
    locationmode="country names",
    size="counts",
    projection="natural earth",
    hover_name="index",
    color="index",
    title="Relative Tweet Counts per Country",
)

# Most popular hashtags
hashtag_counts = df.tweet.str.extractall(r'(\#\w+)')[0].value_counts().reset_index(name='counts')
fig_popular_hashtags = px.bar(
    hashtag_counts[hashtag_counts.counts > 10],
    x='index',
    y='counts',
    title="Most Popular Hashtags"
)


app.layout = html.Div(children=[
    html.H1(children='Twitter Dashboard'),

    html.Div(children='''
        Vizualisations for Twitter Sentiment Data.
    '''),
    html.Div(
        dcc.Graph(
            id='tweets_over_time',
            figure=fig_buckets,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_tweet_sentiment_time_buckets',
            figure=fig_tweet_sentiment_time_buckets,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='pie_overall_sent',
            figure=fig_pie_overall_sent,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='geo_tweet_counts',
            figure=fig_geo_tweet_counts,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_popular_hashtags',
            figure=fig_popular_hashtags,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
