# ==========================================
# STREAMLIT APP FOR NLP TOPIC MODELING
# Python 3.14 Compatible
# File Name: app.py
# ==========================================

import streamlit as st
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

nltk.download('punkt')
nltk.download('stopwords')

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Philippine Government NLP Analyzer",
    page_icon="🇵🇭",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================

lda_model = joblib.load("lda_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==========================================
# STOPWORDS
# ==========================================

stop_words = set(stopwords.words("english"))

# ==========================================
# TEXT PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # join
    cleaned = " ".join(tokens)

    return cleaned

# ==========================================
# GET TOPIC WORDS
# ==========================================

def get_topic_words(model, feature_names, n_top_words=5):

    topics = []

    for topic_idx, topic in enumerate(model.components_):

        top_features = [
            feature_names[i]
            for i in topic.argsort()[-n_top_words:]
        ]

        topics.append(top_features)

    return topics

# ==========================================
# APP TITLE
# ==========================================

st.title("🇵🇭 Philippine Government NLP Topic Modeling")

st.markdown("""
This application analyzes Philippine government related news and predicts possible discussion topics using:

- LDA Topic Modeling
- NLP Text Processing
- Word Frequency Analysis
- Topic Similarity
""")

# ==========================================
# USER INPUT
# ==========================================

user_input = st.text_area(
    "Enter Government Related News or Statement",
    height=200
)

# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button("Analyze Topic"):

    if user_input.strip() == "":

        st.warning("Please enter text first.")

    else:

        # ==========================================
        # PREPROCESS INPUT
        # ==========================================

        cleaned_text = preprocess_text(user_input)

        st.subheader("🧹 Cleaned Text")

        st.write(cleaned_text)

        # ==========================================
        # VECTORIZE
        # ==========================================

        text_vector = vectorizer.transform([cleaned_text])

        # ==========================================
        # TOPIC PREDICTION
        # ==========================================

        topic_distribution = lda_model.transform(text_vector)

        predicted_topic = np.argmax(topic_distribution)

        confidence = np.max(topic_distribution)

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        st.subheader("📌 Predicted Topic")

        st.success(f"Topic #{predicted_topic + 1}")

        st.write(f"Confidence Score: {confidence:.4f}")

        # ==========================================
        # DISPLAY TOPIC WORDS
        # ==========================================

        st.subheader("🔍 Topic Keywords")

        feature_names = vectorizer.get_feature_names_out()

        topics = get_topic_words(
            lda_model,
            feature_names
        )

        for idx, topic_words in enumerate(topics):

            st.write(
                f"Topic {idx + 1}: {', '.join(topic_words)}"
            )

        # ==========================================
        # TOPIC DISTRIBUTION CHART
        # ==========================================

        st.subheader("📊 Topic Distribution")

        fig, ax = plt.subplots()

        topic_labels = [
            f"Topic {i+1}"
            for i in range(len(topic_distribution[0]))
        ]

        ax.bar(
            topic_labels,
            topic_distribution[0]
        )

        ax.set_ylabel("Probability")
        ax.set_title("Topic Probability Distribution")

        st.pyplot(fig)

        # ==========================================
        # WORD CLOUD
        # ==========================================

        st.subheader("☁️ Word Cloud")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(cleaned_text)

        fig_wc, ax_wc = plt.subplots()

        ax_wc.imshow(wordcloud)

        ax_wc.axis("off")

        st.pyplot(fig_wc)

        # ==========================================
        # TOPIC SIMILARITY
        # ==========================================

        st.subheader("🧠 Topic Similarity Matrix")

        similarity_matrix = np.dot(
            topic_distribution,
            topic_distribution.T
        )

        st.write(similarity_matrix)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📚 About Project")

st.sidebar.info("""
This NLP Topic Modeling project focuses on:

- Philippine Government Issues
- Inflation
- Education
- Healthcare
- Transportation
- Elections
- Corruption
- Foreign Policy

Built using:
- Python 3.14
- Streamlit
- Scikit-learn
- NLTK
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Developed using NLP Topic Modeling with LDA"
)
