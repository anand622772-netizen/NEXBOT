from flask import Flask, render_template, request, jsonify
from chatbot import get_response, NexBotTrainer
import os

app = Flask(__name__)

if not os.path.exists("model/tfidf_vectorizer.pkl"):
    NexBotTrainer().train()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"response": "Say something!", "confidence": 0})
    result = get_response(msg)
    return jsonify({"response": result["answer"], "confidence": result["confidence"]})

if __name__ == "__main__":
    app.run(debug=True)
