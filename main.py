import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# モデルとベクタライザーの呼び出し
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# FastAPI appを定義
app = FastAPI()

# リクエストボディの定義
class TextData(BaseModel):
    text: str

# predict_text関数を定義
def predict_text(text: str):
    text_vector = vectorizer.transform([text])
    predictions = model.predict(text_vector)
    predicted_class = predictions[0]
    confidence = np.max(model.decision_function(text_vector))
    return predicted_class, confidence

@app.post("/predict")
async def predict(data: TextData):
    predicted_class, confidence = predict_text(data.text)
    return {"predictions": [{"classification_results": predicted_class, "confidence": confidence}]}
