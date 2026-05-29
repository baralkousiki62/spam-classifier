import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =========================
# NLTK DOWNLOADS (SAFE FOR CLOUD)
# =========================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# =========================
# LOAD MODEL + VECTORIZER
# =========================
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# =========================
# TEXT PREPROCESSING
# =========================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

# =========================
# CUSTOM CSS (CLEAN UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stTextArea textarea {
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    border: none;
}

.result-spam {
    text-align: center;
    color: red;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.result-ham {
    text-align: center;
    color: limegreen;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UI HEADER
# =========================
st.markdown('<div class="title">📩 SMS Spam Classifier</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Detect whether a message is Spam or Not Spam using Machine Learning</div>', unsafe_allow_html=True)

# =========================
# INPUT
# =========================
input_sms = st.text_area("Enter your message here")

# =========================
# PREDICTION
# =========================
if st.button("Predict", key="predict_btn"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result-spam">🚨 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">✅ Not Spam</div>', unsafe_allow_html=True)