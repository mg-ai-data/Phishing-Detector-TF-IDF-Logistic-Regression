# Instalar dependencias (Colab / entorno limpio)
!pip install scikit-learn --quiet

# Código mínimo: detector de phishing (NLP)
import json
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# --- Datos de ejemplo (muy pequeños; reemplazar por dataset real) ---
# 1 = phishing, 0 = legit
train_texts = [
    "Your account has been suspended. Click here to verify: http://fake.link",
    "Important: update your payment information immediately or access will be blocked.",
    "Win a free iPhone! Claim now by providing your details.",
    "Dear user, your invoice is attached. Please see the attached PDF.",
    "Meeting moved to 3pm today, please confirm attendance.",
    "Here is the report from last week's meeting. Thanks.",
    "Reminder: your subscription will renew on 2025-12-15. No action needed.",
    "URGENT: We detected unusual login. Reset password here: http://phish.link"
]
train_labels = [1, 1, 1, 0, 0, 0, 0, 1]

# --- Entrenar modelo (pipeline TF-IDF + regresión logística) ---
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000),
    LogisticRegression(solver='liblinear')
)
model.fit(train_texts, train_labels)

# --- Función de predicción simple ---
def predict_phishing(messages: List[str]):
    """
    Recibe lista de mensajes (strings).
    Devuelve lista de dicts: {text, label, prob_phish}
    """
    probs = model.predict_proba(messages)  # columnas [P(0), P(1)]
    preds = model.predict(messages)
    out = []
    for text, p, pred in zip(messages, probs, preds):
        out.append({
            "text": text,
            "is_phishing": bool(pred),
            "prob_phishing": round(float(p[1]), 4)
        })
    return out

# --- Ejemplo de uso ---
examples = [
    "Please update your billing details here: http://fakebank.example.com",
    "Let's have a short call tomorrow to review the figures."
]

res = predict_phishing(examples)
print(json.dumps(res, indent=2, ensure_ascii=False))

# --- Guardar el modelo para deploy (opcional) ---
import joblib
joblib.dump(model, "phish_model.joblib")
print("Modelo guardado en phish_model.joblib")
