from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "SVM + W2V": {
        "model": joblib.load("D:/data-mining/models/models/svm_model.pkl"),
        "w2v": Word2Vec.load("D:/data-mining/models/models/word2vec.model"),
        "label_encoder": joblib.load("D:/data-mining/models/models/label_encoder.pkl"),
        "type": "w2v",
    },
    "Naive Bayes + TF-IDF": {
        "model": joblib.load("D:/data-mining/models/models2/naive_bayes_model.pkl"),
        "tfidf": joblib.load("D:/data-mining/models/models2/tfidf_vectorizer.pkl"),
        "label_encoder": joblib.load("D:/data-mining/models/models2/label_encoder.pkl"),
        "type": "tfidf",
    },
     "phoBERT": {
        "model": AutoModelForSequenceClassification.from_pretrained("thangtrann/phoBERT_of_thang"),
        "tokenizer": AutoTokenizer.from_pretrained("thangtrann/phoBERT_of_thang"),
        "label_encoder": joblib.load("D:/data-mining/models/models2/label_encoder.pkl"),  
        "type": "bert",
    },
}


class PredictRequest(BaseModel):
    text: str
    model_name: str

@app.post("/predict")
async def predict(req: PredictRequest):
    data = req.text
    model_name = req.model_name

    if model_name not in models:
        return {"error": "Model not found"}

    model_data = models[model_name]
    model_type = model_data['type']

    if model_type == "w2v":
        tokens = simple_preprocess(data)
        vector = get_avg_vector(tokens, model_data['w2v']).reshape(1, -1)
    elif model_type == "tfidf":
        vector = model_data['tfidf'].transform([data])
    elif model_type == "bert":
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        prediction = [predicted_class_id]
    else:
        return {"error": "Unsupported model type"}

    prediction = model_data['model'].predict(vector)
    label = model_data['label_encoder'].inverse_transform(prediction)[0]
    return {"prediction": label}


def get_avg_vector(tokens, model, vector_size=300):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
