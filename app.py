# app.py
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

model = joblib.load("./models/document_classifier_model.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

categories = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
    'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
    'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

st.title("📄 Intelligent Document Classifier (20 Newsgroups)")
st.write("Classify text documents into one of 20 categories using a trained Logistic Regression model.")

user_input = st.text_area("Enter your document text here:")

if st.button("Classify"):
    if user_input.strip():
        clean_text = preprocess_text(user_input)
        vec = vectorizer.transform([clean_text])
        prediction = model.predict(vec)[0]
        st.success(f"**Predicted Category:** {categories[prediction]}")
    else:
        st.warning("Please enter some text before classifying.")
