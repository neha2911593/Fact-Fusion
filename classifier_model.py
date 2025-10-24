from transformers import pipeline

# Load the DistilBERT model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to classify a claim
def classify_claim(claim):
    result = classifier(claim)[0]
    label = "True" if result['label'] == "POSITIVE" else "False"
    confidence = result['score'] * 100
    return label, confidence
