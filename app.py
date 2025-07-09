import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from googletrans import Translator
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize translator and sentiment analyzer
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

# Load data function
@st.cache_data
def load_data():
    return pd.read_csv('amazon_reviews_file-July 9 2025.csv')

# Function to translate reviews (if needed)
def translate_review(review, target_language='en'):
    try:
        translated = translator.translate(review, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return review

# Function to analyze sentiment
def analyze_sentiment(review):
    sentiment = analyzer.polarity_scores(review)
    return sentiment['compound']

# Main app
st.title("Amazon Reviews Dashboard")

# Load data
df = load_data()

# Sidebar for filters
st.sidebar.header("Filters")
year_filter = st.sidebar.slider("Select Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=(int(df['year'].min()), int(df['year'].max())))
filtered_df = df[(df['year'] >= year_filter[0]) & (df['year'] <= year_filter[1])]

# Sentiment Analysis
st.header("Sentiment Analysis")
if not filtered_df.empty:
    filtered_df['sentiment_score'] = filtered_df['reviewText'].apply(analyze_sentiment)
    avg_sentiment = filtered_df['sentiment_score'].mean()
    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # Sentiment Distribution
    fig1 = px.histogram(filtered_df, x='sentiment_score', nbins=20, title="Sentiment Score Distribution")
    st.plotly_chart(fig1)

# Word Cloud
st.header("Word Cloud")
if not filtered_df.empty:
    stop_words = set(stopwords.words('english'))
    text = ' '.join(review for review in filtered_df['reviewText'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, min_font_size=10).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Trend Analysis (using ARIMA for example)
st.header("Review Trends")
if not filtered_df.empty:
    # Aggregate reviews by year
    trend_data = filtered_df.groupby('year').size().reset_index(name='review_count')
    # Fit ARIMA model (example with minimal data)
    model = ARIMA(trend_data['review_count'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=5)
    st.line_chart(trend_data.set_index('year').join(pd.DataFrame({'forecast': forecast}, index=range(trend_data['year'].max() + 1, trend_data['year'].max() + 6))))

# Comment out gensim-related code
'''
import gensim
from gensim import corpora

def get_topics(df):
    # Tokenize reviews
    tokenized_reviews = [word_tokenize(review.lower()) for review in df['reviewText']]
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_reviews)
    corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]
    # Train LDA model
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    topics = lda_model.print_topics()
    return topics

st.header("Topic Modeling")
if not filtered_df.empty:
    topics = get_topics(filtered_df)
    for topic in topics:
        st.write(topic)
'''

# Additional Stats
st.header("Additional Statistics")
if not filtered_df.empty:
    avg_score = filtered_df['overall score - 1 is bad and 5 is excellent'].mean()
    st.metric("Average Rating", f"{avg_score:.2f}")
