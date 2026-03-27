import os
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from dataset import training_data

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

MODEL_DIR   = "model"
VECTOR_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess_text(text):
    text   = text.lower()
    text   = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalpha()]
    return " ".join(tokens)


class NexBotTrainer:
    def __init__(self):
        self.questions  = [p[0] for p in training_data]
        self.answers    = [p[1] for p in training_data]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    def train(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        clean_qs     = [preprocess_text(q) for q in self.questions]
        tfidf_matrix = self.vectorizer.fit_transform(clean_qs)
        with open(VECTOR_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
        print(f"Training done. Vocab: {len(self.vectorizer.vocabulary_)} terms")
        return self.vectorizer, tfidf_matrix


class NexBotPredictor:
    def __init__(self):
        self.questions = [p[0] for p in training_data]
        self.answers   = [p[1] for p in training_data]
        self.vectorizer, self.tfidf_matrix = self._load()

    def _load(self):
        if not os.path.exists(VECTOR_PATH):
            return NexBotTrainer().train()
        with open(VECTOR_PATH, "rb") as f:
            v = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            m = pickle.load(f)
        return v, m

    def predict(self, user_input):
        if not user_input.strip():
            return {"answer": "Please type something!", "confidence": 0.0}
        vec   = self.vectorizer.transform([preprocess_text(user_input)])
        sims  = cosine_similarity(vec, self.tfidf_matrix).flatten()
        idx   = int(np.argmax(sims))
        score = float(sims[idx])
        if score < 0.15:
            return {"answer": "Not sure about that. Ask me about IT, ML, programming or college topics!", "confidence": round(score * 100, 2)}
        return {"answer": self.answers[idx], "confidence": round(score * 100, 2)}


_predictor = None

def get_response(user_input):
    global _predictor
    if _predictor is None:
        _predictor = NexBotPredictor()
    return _predictor.predict(user_input)
