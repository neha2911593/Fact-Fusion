from models import nli_model  # shared model


def verify_claim(claim, evidence):
    # Format input properly for NLI models
    text = f"premise: {evidence}\nhypothesis: {claim}"

    result = nli_model(text)[0]
    nli_label = result["label"]
    score = round(result["score"] * 100, 2)

    # Convert NLI labels into human-friendly truth labels
    if nli_label == "ENTAILMENT":
        final_label = "True"
    elif nli_label == "CONTRADICTION":
        final_label = "False"
    else:
        final_label = "Uncertain"

    return final_label, score
