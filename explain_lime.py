from lime.lime_text import LimeTextExplainer
from transformers import pipeline
import numpy as np

# Load model once
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
explainer = LimeTextExplainer(class_names=["False", "True"])

def explain_with_lime(text):
    def predict_proba(texts):
        # texts is list of strings → run model on each
        results = model(texts)
        # Convert POSITIVE → [prob_false, prob_true], NEGATIVE → [prob_true, prob_false]
        probs = []
        for r in results:
            score = r['score']
            if r['label'] == 'POSITIVE':
                probs.append([1 - score, score])
            else:
                probs.append([score, 1 - score])
        return np.array(probs)
    
    try:
        exp = explainer.explain_instance(text, predict_proba, num_features=8)
        return exp.as_html()
    except Exception as e:
        return f"<p style='color:red;'>LIME explanation failed: {str(e)}</p>"