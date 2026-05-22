# ==========================================
# SIMPLE STREAMLIT NLP TOPIC PREDICTOR
# ==========================================

import streamlit as st
import joblib
import re
import nltk
import numpy as np

from nltk.corpus import stopwords

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

nltk.download('stopwords', quiet=True)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="NLP Topic Predictor",
    page_icon="🇵🇭",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

lda_model = joblib.load("lda_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# ==========================================
# STOPWORDS
# ==========================================

stop_words = set(stopwords.words("english"))

# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize using split
    tokens = text.split()

    # remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # join cleaned text
    cleaned = " ".join(tokens)

    return cleaned

# ==========================================
# TOPIC LABELS
# ==========================================

topic_labels = {
    0: "Government & Public Services",
    1: "Politics & National Issues"
}

# ==========================================
# UI
# ==========================================

st.title("🇵🇭 Philippine Government Topic Predictor")

user_input = st.text_area(
    "Enter a statement or news article:",
    height=200
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Topic"):

    if user_input.strip() == "":

        st.warning("Please enter text.")

    else:

        # preprocess
        cleaned_text = preprocess_text(user_input)

        # vectorize
        text_vector = vectorizer.transform([cleaned_text])

        # predict
        topic_distribution = lda_model.transform(text_vector)

        predicted_topic = np.argmax(topic_distribution)

        predicted_label = topic_labels.get(
            predicted_topic,
            "Unknown Topic"
        )

        # display only topic
        st.success(
            f"Predicted Topic: {predicted_label}"
        )
