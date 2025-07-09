import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import f_oneway, pearsonr
from statsmodels.tsa.arima.model import ARIMA
import nltk
import re
import io
import base64

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tools
translator = Translator()
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('amazon_reviews_file-July 9 2025.csv')
    df['review_date'] = pd.to_datetime(df['review_date'], format='%d-%m-%Y')
    df['year'] = df['year'].astype(int)
    df['month'] = df['review_date'].dt.month
    df['reviewText'] = df['reviewText'].fillna('')
    df['summary'] = df['summary'].fillna('')
    return df

@st.cache_data
def preprocess_data(df):
    # Translate non-English reviews
    def translate_text(text):
        try:
            detected = translator.detect(text)
            if detected.lang != 'en' and text.strip():
                translated = translator.translate(text, dest='en').text
                return translated
            return text
        except:
            return text

    df['translated_reviewText'] = df['reviewText'].apply(translate_text)
    
    # Sentiment analysis
    def get_sentiment(text):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df['translated_reviewText'].apply(get_sentiment)
    
    # Fraud detection
    def detect_fraud(row, df):
        review_text = row['translated_reviewText'].lower()
        reviewer = row['reviewerName']
        review_date = row['review_date']
        
        # Duplicate reviews
        duplicates = df[df['translated_reviewText'].str.lower() == review_text]
        is_duplicate = len(duplicates) > 1
        
        # Multiple reviews by same user
        user_reviews = df[df['reviewerName'] == reviewer]
        multiple_reviews = len(user_reviews) > 1
        
        # Short or gibberish reviews
        words = word_tokenize(review_text)
        unique_words = len(set(words))
        is_short = len(words) < 5 or unique_words < 5
        
        # Excessive uppercase
        uppercase_ratio = sum(1 for c in review_text if c.isupper()) / (len(review_text) + 1)
        is_uppercase = uppercase_ratio > 0.5
        
        if is_duplicate or multiple_reviews or is_short or is_uppercase:
            reasons = []
            if is_duplicate:
                reasons.append("Duplicate review")
            if multiple_reviews:
                reasons.append("Multiple reviews by user")
            if is_short:
                reasons.append("Short/gibberish review")
            if is_uppercase:
                reasons.append("Excessive uppercase")
            return 'Yes', '; '.join(reasons)
        return 'No', ''
    
    df[['is_fraud', 'fraud_reason']] = df.apply(lambda row: pd.Series(detect_fraud(row, df)), axis=1)
    
    return df

# Apply filters
def apply_filters(df, rating, year, month, sentiment, fraud, keyword):
    filtered = df.copy()
    if rating:
        filtered = filtered[filtered['overall score - 1 is bad and 5 is excellent'].isin(rating)]
    if year:
        filtered = filtered[filtered['year'].isin(year)]
    if month:
        filtered = filtered[filtered['month'].isin(month)]
    if sentiment:
        filtered = filtered[filtered['sentiment'].isin(sentiment)]
    if fraud:
        filtered = filtered[filtered['is_fraud'].isin(fraud)]
    if keyword:
        filtered = filtered[filtered['translated_reviewText'].str.contains(keyword, case=False, na=False)]
    return filtered

# Generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# LDA Topic Modeling
def get_topics(texts, num_topics=5):
    tokens = [word_tokenize(text.lower()) for text in texts]
    tokens = [[word for word in doc if word not in stop_words and word.isalpha()] for doc in tokens]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda.print_topics()
    return topics

# Semantic Clustering
def semantic_clustering(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X.toarray())
    return clustering.labels_

# Statistical Analysis
def statistical_analysis(df):
    # ANOVA: Rating vs Sentiment
    groups = [df[df['sentiment'] == s]['overall score - 1 is bad and 5 is excellent'] for s in df['sentiment'].unique()]
    anova_result = f_oneway(*groups)
    
    # Correlation: Review length vs Rating
    df['review_length'] = df['translated_reviewText'].apply(len)
    corr, p_value = pearsonr(df['review_length'], df['overall score - 1 is bad and 5 is excellent'])
    
    return anova_result, corr, p_value

# Trend Forecasting
def forecast_trends(df):
    df['date'] = pd.to_datetime(df['review_date'])
    monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
    model = ARIMA(monthly_counts.values, order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=12)
    return monthly_counts.index.to_timestamp(), monthly_counts.values, forecast

# Streamlit App
st.set_page_config(page_title="Amazon Reviews Dashboard", layout="wide")
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    .stApp { background-color: #f9fafb; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .header { font-size: 2.5rem; font-weight: bold; color: #1f2937; text-align: center; margin-bottom: 20px; }
    .subheader { font-size: 1.5rem; font-weight: 600; color: #374151; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Amazon Reviews Interactive Dashboard</div>', unsafe_allow_html=True)

# Load data
df = load_data()
df = preprocess_data(df)

# Sidebar Filters
st.sidebar.header("Filters")
rating_options = sorted(df['overall score - 1 is bad and 5 is excellent'].unique())
selected_rating = st.sidebar.multiselect("Star Rating", rating_options, default=rating_options)
year_options = sorted(df['year'].unique())
selected_year = st.sidebar.multiselect("Year", year_options, default=year_options)
month_options = sorted(df['month'].unique())
selected_month = st.sidebar.multiselect("Month", month_options, default=month_options)
sentiment_options = df['sentiment'].unique()
selected_sentiment = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)
fraud_options = df['is_fraud'].unique()
selected_fraud = st.sidebar.multiselect("Fraud Status", fraud_options, default=fraud_options)
keyword = st.sidebar.text_input("Keyword Search")

# Apply filters
filtered_df = apply_filters(df, selected_rating, selected_year, selected_month, selected_sentiment, selected_fraud, keyword)

# Tabs
tabs = st.tabs(["Overview", "Verbatim Analysis", "Sentiment Analysis", "Fraud Detection", "Advanced Analytics"])

# Overview Tab
with tabs[0]:
    st.markdown('<div class="subheader">Overview Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Star Rating Distribution
        fig = px.histogram(filtered_df, x='overall score - 1 is bad and 5 is excellent', title="Star Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment Distribution
        fig = px.pie(filtered_df, names='sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Review Trends Over Time
    trends = filtered_df.groupby(['year', 'month']).size().reset_index(name='count')
    trends['date'] = pd.to_datetime(trends[['year', 'month']].assign(day=1))
    fig = px.line(trends, x='date', y='count', title="Review Trends Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic Executive Summary
    summary = f"""
    **Executive Summary**  
    - **Total Reviews**: {len(filtered_df)}  
    - **Average Rating**: {filtered_df['overall score - 1 is bad and 5 is excellent'].mean():.2f}  
    - **Sentiment Breakdown**: {', '.join([f'{s}: {len(filtered_df[filtered_df['sentiment'] == s])}' for s in filtered_df['sentiment'].unique()])}  
    - **Key Insight**: {f'Most reviews are positive, indicating high satisfaction.' if filtered_df['sentiment'].value_counts().idxmax() == 'Positive' else 'Mixed sentiments suggest areas for improvement.'}
    """
    st.markdown(summary, unsafe_allow_html=True)

# Verbatim Analysis Tab
with tabs[1]:
    st.markdown('<div class="subheader">Verbatim Analysis</div>', unsafe_allow_html=True)
    
    # Word Cloud
    text = ' '.join(filtered_df['translated_reviewText'])
    fig = generate_wordcloud(text)
    st.pyplot(fig)
    
    # Keyword Frequency
    words = word_tokenize(text.lower())
    words = [w for w in words if w not in stop_words and w.isalpha()]
    word_freq = Counter(words).most_common(10)
    fig = px.bar(x=[w[0] for w in word_freq], y=[w[1] for w in word_freq], title="Top 10 Keywords")
    st.plotly_chart(fig, use_container_width=True)
    
    # LDA Topic Modeling
    topics = get_topics(filtered_df['translated_reviewText'])
    st.write("**Key Topics**")
    for topic in topics:
        st.write(f"Topic {topic[0]}: {topic[1]}")
    
    # Semantic Clustering
    clusters = semantic_clustering(filtered_df['translated_reviewText'])
    filtered_df['cluster'] = clusters
    fig = px.histogram(filtered_df, x='cluster', title="Semantic Clusters")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis Tab
with tabs[2]:
    st.markdown('<div class="subheader">Sentiment Analysis</div>', unsafe_allow_html=True)
    
    # Sentiment by Star Rating
    fig = px.box(filtered_df, x='sentiment', y='overall score - 1 is bad and 5 is excellent', title="Sentiment by Star Rating")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment by Date
    sentiment_trends = filtered_df.groupby(['year', 'month', 'sentiment']).size().unstack().fillna(0).reset_index()
    sentiment_trends['date'] = pd.to_datetime(sentiment_trends[['year', 'month']].assign(day=1))
    fig = go.Figure()
    for sentiment in sentiment_trends.columns[2:-1]:
        fig.add_trace(go.Scatter(x=sentiment_trends['date'], y=sentiment_trends[sentiment], name=sentiment))
    fig.update_layout(title="Sentiment Trends Over Time")
    st.plotly_chart(fig, use_container_width=True)

# Fraud Detection Tab
with tabs[3]:
    st.markdown('<div class="subheader">Fraud Detection</div>', unsafe_allow_html=True)
    
    # Fraud Summary
    fraud_counts = filtered_df['is_fraud'].value_counts()
    fig = px.pie(values=fraud_counts.values, names=fraud_counts.index, title="Fraudulent vs Non-Fraudulent Reviews")
    st.plotly_chart(fig, use_container_width=True)
    
    # Fraud Reasons
    fraud_df = filtered_df[filtered_df['is_fraud'] == 'Yes']
    reasons = fraud_df['fraud_reason'].str.split('; ').explode().value_counts()
    fig = px.bar(x=reasons.index, y=reasons.values, title="Fraud Reasons")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display Suspect Reviews
    st.write("**Suspect Reviews**")
    st.dataframe(fraud_df[['reviewerName', 'translated_reviewText', 'fraud_reason']])

# Advanced Analytics Tab
with tabs[4]:
    st.markdown('<div class="subheader">Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Review Length vs Rating
    filtered_df['review_length'] = filtered_df['translated_reviewText'].apply(len)
    fig = px.scatter(filtered_df, x='review_length', y='overall score - 1 is bad and 5 is excellent', title="Review Length vs Rating")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment vs Rating Inconsistency
    inconsistent = filtered_df[((filtered_df['sentiment'] == 'Positive') & (filtered_df['overall score - 1 is bad and 5 is excellent'] < 3)) |
                              ((filtered_df['sentiment'] == 'Negative') & (filtered_df['overall score - 1 is bad and 5 is excellent'] > 3))]
    st.write(f"**Inconsistent Reviews**: {len(inconsistent)}")
    st.dataframe(inconsistent[['reviewerName', 'translated_reviewText', 'sentiment', 'overall score - 1 is bad and 5 is excellent']])
    
    # Statistical Analysis
    anova_result, corr, p_value = statistical_analysis(filtered_df)
    st.write(f"**ANOVA (Rating vs Sentiment)**: F={anova_result.statistic:.2f}, p={anova_result.pvalue:.4f}")
    st.write(f"**Correlation (Review Length vs Rating)**: r={corr:.2f}, p={p_value:.4f}")
    
    # Trend Forecasting
    dates, counts, forecast = forecast_trends(filtered_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=counts, name='Historical'))
    fig.add_trace(go.Scatter(x=pd.date_range(start=dates[-1], periods=13, freq='M')[1:], y=forecast, name='Forecast'))
    fig.update_layout(title="Review Volume Forecast")
    st.plotly_chart(fig, use_container_width=True)

# Download CSV
csv = filtered_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="enhanced_amazon_reviews.csv" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Download Enhanced CSV</a>'
st.markdown(href, unsafe_allow_html=True)
