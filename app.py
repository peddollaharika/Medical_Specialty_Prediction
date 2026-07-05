import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only the first time)
nltk.download("punkt")
nltk.download("stopwords")

# -------------------------------
# Load Model and Vectorizer
# -------------------------------

model = joblib.load("medical_specialty_model_nb.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# Text Preprocessing Function
# -------------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = " ".join(tokens)

    return cleaned_text

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Medical Specialty Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Specialty Prediction")

st.markdown(
"""
Predict the **Medical Specialty** from a doctor's medical transcription using
Machine Learning and Natural Language Processing.
"""
)

user_input = st.text_area(
    "Enter Medical Transcription",
    height=250,
    placeholder="Paste medical transcription here..."
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter medical transcription.")

    else:

        cleaned = preprocess_text(user_input)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        st.success(f"Predicted Medical Specialty: **{prediction[0]}**")