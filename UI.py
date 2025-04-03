import os          # per eseguire: streamlit run UI.py
import sys
sys.path.append(os.path.abspath('..'))
from src import config
import streamlit as st
import pickle

# Load th emodel and vectorizer
with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "wb") as file:
        model= pickle.load(file)

with open(f"(config.MODELS_PATH)vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

st.tile("Text classification")

# Text input
user_input = st.text_area("Enter text to classify", "")

# Predict when button is clicked
if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            # Transform input and predict
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            if prediction == 'positive':
                st.success(f"Predicted class: {prediction}")
            elif prediction == 'negative':
                st.warning(f"Predicted class: {prediction}")
