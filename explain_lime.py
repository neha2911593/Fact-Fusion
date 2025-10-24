import lime
from lime.lime_text import LimeTextExplainer
from transformers import pipeline

explainer = LimeTextExplainer(class_names=["False", "True"])
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def explain_with_lime(text):
    pred_fn = lambda x: [d["score"] for d in model(x)]
    exp = explainer.explain_instance(text, pred_fn, num_features=8)
    return exp.as_html()
