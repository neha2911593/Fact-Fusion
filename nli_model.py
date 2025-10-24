from transformers import pipeline

nli_model = pipeline("text-classification", model="roberta-large-mnli")

def verify_claim(claim, evidence):
    result = nli_model(f"{claim} </s> {evidence}")[0]
    return result['label']
