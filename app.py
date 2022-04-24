from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from create_wordcloud import *
from data_manipulations import *

app = Dash(__name__)

df = pd.read_csv("tweets_annotated.1650575029.formulaone.csv")
# df = pd.read_csv("tweets_annotated.1650577206.elonmusk.csv")
df = prep_text(df)

# Line - Tweets Over Time Buckets
tweet_df_buckets = df.groupby(pd.Grouper(
    key='created_at', freq='60Min', convention='start')).size()
df_t = pd.DataFrame(list(tweet_df_buckets.items()),
                    columns=['created_at', 'counts'])

fig_buckets = px.line(
    df_t,
    x="created_at",
    y="counts",
    title="Tweet Counts in 60 minute buckets",
    labels={
        'created_at': 'Time',
        'counts': 'Count'
    }
)

# Lines - Tweet Sentiment Counts in 60 minute Time Buckets
tweet_sentiment_time_buckets = df.groupby([
    'sentiment',
    pd.Grouper(
        key='created_at', freq='60Min', convention='start')
]).size()
df_t_s = tweet_sentiment_time_buckets.reset_index(name='counts')

fig_tweet_sentiment_time_buckets = px.line(
    df_t_s,
    line_group="sentiment",
    color="sentiment",
    x="created_at",
    y="counts",
    title="Tweet Sentiment Counts in 60 minute Time Buckets",
    labels={
        'created_at': 'Time',
        'sentiment': 'Sentiment',
        'counts': 'Count'
    }
)

# Pie - Sentiment
df_sent_counts = df['sentiment'].value_counts().reset_index(name='counts')
fig_pie_overall_sent = px.pie(
    df_sent_counts,
    values='counts',
    names='index',
    title='Overall Sentiment Pie Chart',
    hole=0.2,
    labels={
        'index': 'Sentiment',
        'counts': 'Count'
    }
)

# Chloropleth - Median Tweet Sentiment by Country
tweet_most_common_sentiment_by_country = df.groupby(
    'country')['sentiment_numeric'].mean().reset_index(name='sentiment_mean')
# print(tweet_most_common_sentiment_by_country)
fig_chloro_average_sentiment = px.choropleth(
    tweet_most_common_sentiment_by_country,
    locations="country",
    locationmode="country names",
    projection="natural earth",
    hover_name="country",
    color="sentiment_mean",
    title="Sentiment Positivity by Country",
    labels={
        'country': 'Country',
        'sentiment_mean': 'Sentiment Positivity',
        'counts': 'Count'
    },
    width=1500,
    height=800
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
    title="Relative Tweet Count per Country",
    labels={
        'index': 'Country',
        'counts': 'Count'
    },
    width=1500,
    height=800
)

# Most popular hashtags
hashtag_counts = df.tweet.str.extractall(
    r'(\#\w+)')[0].value_counts().reset_index(name='counts')
fig_popular_hashtags = px.bar(
    hashtag_counts[hashtag_counts.counts > 10][:10],
    x='index',
    y='counts',
    color='counts',
    title="Most Popular Hashtags",
    labels={
        'index': 'Hashtag',
        'counts': 'Count'
    }
)

# Wordcloud
fig_worldcloud = px.imshow(
    create_wordcloud(df['normalized_tweet']),
    title="Wordcloud"
)

# Bar - Mean Tweet Sentiment by Platform
most_frequent_platforms = df['source'].value_counts()[:10].index.tolist()
tweet_most_common_sentiment_by_platform = df[df['source'].isin(most_frequent_platforms)].groupby(
    'source')['sentiment_numeric'].mean().reset_index(name='sentiment_mean')
fig_tweet_most_common_sentiment_by_platform = px.bar(
    tweet_most_common_sentiment_by_platform.sort_values(by=['sentiment_mean']),
    x='source',
    y='sentiment_mean',
    color='source',
    title="Average Sentiment Positivity by Platform",
    labels={
        'source': 'Platform',
        'sentiment_mean': 'Sentiment Positivity',
    }
)

app.layout = html.Div(children=[
    html.H1(children='Twitter Dashboard'),

    html.Div(children='''
        Vizualisations for Twitter Sentiment Data.
        Dataset consists of tweets search for by term `Formula One`
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
            id='fig_tweet_sentiment_time_buckets',
            figure=fig_tweet_sentiment_time_buckets,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_popular_hashtags',
            figure=fig_popular_hashtags,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_wordcloud',
            figure=fig_worldcloud,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_tweet_most_common_sentiment_by_platform',
            figure=fig_tweet_most_common_sentiment_by_platform,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='fig_chloro_average_sentiment',
            figure=fig_chloro_average_sentiment,
            style={'width': 'auto'}
        ), style={'display': 'inline-block'}
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
