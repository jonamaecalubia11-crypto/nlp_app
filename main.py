# ==========================================
# NLP TOPIC MODELING PROJECT
# Python 3.14 Compatible
# ==========================================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF
)

from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================
# LOAD DATASET
# ==========================================

documents = [

    "The Philippine government continues to address inflation and rising food prices across the country.",

    "Senators discussed the national budget allocation for education and healthcare programs.",

    "The Department of Transportation announced new railway projects to reduce traffic congestion in Metro Manila.",

    "Farmers are requesting more government support due to increasing fertilizer prices and low crop income.",

    "The West Philippine Sea issue remains a major concern in the country's foreign policy discussions.",

    "Students demand better internet access and improved public education facilities.",

    "The government launched additional healthcare programs for low income families.",

    "Unemployment rates increased as several businesses reduced their workforce.",

    "Political analysts discussed possible candidates for the upcoming national elections.",

    "Citizens expressed concerns about corruption and transparency in public offices."

]

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(stopwords.words("english"))

processed_docs = []

for doc in documents:

    # convert to lowercase
    doc = doc.lower()

    # remove punctuation
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)

    # tokenize
    tokens = word_tokenize(doc)

    # remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # join tokens
    cleaned = " ".join(tokens)

    processed_docs.append(cleaned)

print("\n========== CLEANED DOCUMENTS ==========\n")

for doc in processed_docs:
    print(doc)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(processed_docs)

print(X.shape)

print(X)

X

from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(
    n_components=2,   # number of topics
    random_state=42
)

lda_model.fit(X)

print("\n========== LDA TOPICS ==========\n")

lda_feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda_model.components_):

    top_words = [
        lda_feature_names[i]
        for i in topic.argsort()[-5:]
    ]

    print(f"Topic {topic_idx + 1}:")
    print(top_words)
    print()

lda_perplexity = lda_model.perplexity(
    X
)

print("\n========== MODEL EVALUATION ==========\n")

print(f"LDA Perplexity Score: {lda_perplexity}")

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_docs)

word_counts = X.sum(axis=0).A1
words = vectorizer.get_feature_names_out()

top_indices = word_counts.argsort()[-10:][::-1]

top_words = [words[i] for i in top_indices]
top_counts = [word_counts[i] for i in top_indices]

plt.figure()
plt.bar(top_words, top_counts)
plt.xticks(rotation=45)
plt.title("Top 10 Most Frequent Terms")
plt.show()

import numpy as np

# Calculate simple topic coherence using topic-document distributions
print("\n========== COHERENCE SCORES ==========\n")

# LDA Coherence: Average topic distribution entropy
lda_topics = lda_model.transform(X)
lda_entropy = -np.sum(lda_topics * np.log(lda_topics + 1e-10), axis=1)
lda_coherence = 1 - (np.mean(lda_entropy) / np.log(lda_model.n_components))

print(f"LDA Coherence Score: {lda_coherence:.4f}")


lda_topics = lda_model.transform(X)

similarity = cosine_similarity(lda_topics)

print("\n========== TOPIC SIMILARITY ==========\n")

print(similarity)

import joblib

# Create and fit tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(processed_docs)

joblib.dump(
    lda_model,
    "lda_model.pkl"
)

joblib.dump(
    vectorizer,
    "count_vectorizer.pkl"
)

joblib.dump(
    tfidf_vectorizer,
    "tfidf_vectorizer.pkl"
)

print("\nModels saved successfully!")