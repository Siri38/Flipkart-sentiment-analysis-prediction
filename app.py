# import numpy as np
# import pandas as pd
# import re
# import emoji
# from textblob import TextBlob
# import nltk
# from nltk.tokenize import word_tokenize,sent_tokenize
# nltk.download('punkt')
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.metrics import accuracy_score,f1_score
import streamlit as st
import joblib

result = None
st.title("Flipkart based Review Detection using ML")
text = st.text_input("Enter the text")

model = joblib.load(r'D:\Internship\Sentiment_analysis\naive_bayes.pkl')

if st.button("Submit")==True:
    result = model.predict([text])[0]
    st.write(result)

if result == 'Positive':
    st.image("https://as1.ftcdn.net/v2/jpg/04/00/44/12/1000_F_400441282_YdhdOgOryXWZ88umIGD4oRtfvoWfFUYe.jpg")
elif result == 'Negative':
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Icon_no.svg/768px-Icon_no.svg.png")