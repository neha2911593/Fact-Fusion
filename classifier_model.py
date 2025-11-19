from models import nli_model  # shared model

def classify_claim(claim):
    result = nli_model(claim)[0]

    if result['label'] == "ENTAILMENT":
        label = "True"
    elif result['label'] == "CONTRADICTION":
        label = "False"
    else:
        label = "Uncertain"

    confidence = round(result['score'] * 100, 2)
    return label, confidence
