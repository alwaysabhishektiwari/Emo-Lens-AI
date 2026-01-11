import streamlit as st
import joblib
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- PAGE CONFIG ---
st.set_page_config(page_title="EmoLens | Emotion AI", page_icon="🧠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    .stTextArea textarea { border-radius: 10px; border: 2px solid #6c757d; }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .emotion-text { font-size: 30px; font-weight: bold; color: #4A90E2; }
    </style>
    """, unsafe_allow_html=True)

# --- PREPROCESSING [ Notebook cells 109-121  ] ---
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(txt):
    txt = txt.lower() ##
    txt = txt.translate(str.maketrans('', '', string.punctuation)) #
    txt = ''.join([i for i in txt if not i.isdigit()]) #
    txt = ''.join([i for i in txt if i.isascii()]) #
    words = word_tokenize(txt) #
    cleaned = [w for w in words if w not in stop_words] #
    return ' '.join(cleaned)

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

emo_dict = {
    0: ("Sadness", "😢"), 1: ("Anger", "😠"), 2: ("Love", "❤️"), 
    3: ("Surprise", "😯"), 4: ("Fear", "😨"), 5: ("Joy", "😊")
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("This model uses Logistic Regression with TF-IDF Vectorization.")
    if st.button("Clear Cache"):
        st.cache_resource.clear()
    st.divider()
    st.markdown("Created By Abhishek Tiwari using Scikit-Learn & Streamlit")

# --- MAIN USER INTER ---
st.title("🧠 EmoLens: Emotion Analysis")
st.markdown("Analyze the underlying sentiment of your text using Natural Language Processing.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter text to analyze:", height=150, placeholder="Type how you feel...")
    predict_btn = st.button("Analyze Emotion 🔍", use_container_width=True)

if predict_btn:
    if user_input.strip():
        model, vectorizer = load_assets()
        
        #  input
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        
        # Prediction
        pred_idx = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        label, icon = emo_dict[pred_idx]
        
        with col2:
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="font-size: 50px;">{icon}</p>
                    <p>Detected Emotion:</p>
                    <p class="emotion-text">{label}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence Scores
        st.divider()
        st.subheader("📊 Confidence Analysis")
        cols = st.columns(6)
        for i, (idx, (name, emoji)) in enumerate(emo_dict.items()):
            with cols[i]:
                st.metric(label=f"{emoji} {name}", value=f"{probs[i]*100:.1f}%")
                st.progress(float(probs[i]))
    else:
        st.error("Please enter text before analyzing.")