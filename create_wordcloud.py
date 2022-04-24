import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(series):
  wordcloud = WordCloud(width=800, height=400, margin=2).generate(' '.join(series))
  return wordcloud.to_image()