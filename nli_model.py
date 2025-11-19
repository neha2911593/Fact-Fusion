from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load RoBERTa model for NLI
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_claim(claim: str):
    """
    Classify a claim as True, False, or Uncertain using zero-shot classification
    
    Args:
        claim: The claim text to classify
        
    Returns:
        tuple: (classification_label, confidence_score)
    """
    # Define hypothesis templates
    hypothesis_true = f"This statement is scientifically accurate: {claim}"
    hypothesis_false = f"This statement is scientifically false: {claim}"
    
    # Tokenize inputs
    inputs_true = tokenizer(claim, hypothesis_true, return_tensors="pt", truncation=True, max_length=512)
    inputs_false = tokenizer(claim, hypothesis_false, return_tensors="pt", truncation=True, max_length=512)
    
    # Get predictions
    with torch.no_grad():
        outputs_true = model(**inputs_true)
        outputs_false = model(**inputs_false)
    
    # Apply softmax to get probabilities
    probs_true = torch.softmax(outputs_true.logits, dim=1)
    probs_false = torch.softmax(outputs_false.logits, dim=1)
    
    # Get entailment probabilities (index 2 is entailment in RoBERTa-MNLI)
    entailment_true = probs_true[0][2].item()
    entailment_false = probs_false[0][2].item()
    
    # Decision logic with adjusted thresholds
    TRUE_THRESHOLD = 0.50
    FALSE_THRESHOLD = 0.50
    CONFIDENCE_MARGIN = 0.10
    
    # If true score is higher and above threshold
    if entailment_true > entailment_false + CONFIDENCE_MARGIN and entailment_true > TRUE_THRESHOLD:
        return "True", entailment_true * 100  # Convert to percentage
    
    # If false score is higher and above threshold
    elif entailment_false > entailment_true + CONFIDENCE_MARGIN and entailment_false > FALSE_THRESHOLD:
        return "False", entailment_false * 100
    
    # If scores are similar or both below threshold
    else:
        max_confidence = max(entailment_true, entailment_false)
        return "Uncertain", max_confidence * 100


def verify_with_evidence(claim: str, evidence: str):
    """
    Verify a claim against retrieved evidence using NLI
    
    Args:
        claim: The claim to verify
        evidence: The evidence text retrieved from sources
        
    Returns:
        tuple: (verification_label, confidence_score as percentage)
    """
    # Truncate evidence if too long (keep first 500 words)
    evidence_words = evidence.split()
    if len(evidence_words) > 500:
        evidence = " ".join(evidence_words[:500]) + "..."
    
    # Tokenize the claim-evidence pair
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(outputs.logits, dim=1)
    
    # RoBERTa-MNLI outputs: [contradiction, neutral, entailment]
    contradiction_score = probs[0][0].item()
    neutral_score = probs[0][1].item()
    entailment_score = probs[0][2].item()
    
    # Adjusted thresholds for better classification
    ENTAILMENT_THRESHOLD = 0.45
    CONTRADICTION_THRESHOLD = 0.45
    NEUTRAL_MARGIN = 0.15
    
    # Get the highest score
    max_score = max(entailment_score, contradiction_score, neutral_score)
    
    # Decision logic with improved sensitivity
    if entailment_score == max_score:
        if entailment_score > ENTAILMENT_THRESHOLD:
            # Check if neutral is close to entailment
            if neutral_score > entailment_score - NEUTRAL_MARGIN:
                return "Uncertain", neutral_score * 100
            return "True", entailment_score * 100
        else:
            return "Uncertain", entailment_score * 100
    
    elif contradiction_score == max_score:
        if contradiction_score > CONTRADICTION_THRESHOLD:
            # Check if neutral is close to contradiction
            if neutral_score > contradiction_score - NEUTRAL_MARGIN:
                return "Uncertain", neutral_score * 100
            return "False", contradiction_score * 100
        else:
            return "Uncertain", contradiction_score * 100
    
    else:  # neutral_score is highest
        return "Uncertain", neutral_score * 100


def get_nli_scores(claim: str, evidence: str):
    """
    Get detailed NLI scores for debugging and analysis
    
    Args:
        claim: The claim to verify
        evidence: The evidence text
        
    Returns:
        dict: Dictionary containing all three scores
    """
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    
    return {
        "contradiction": round(probs[0][0].item(), 4),
        "neutral": round(probs[0][1].item(), 4),
        "entailment": round(probs[0][2].item(), 4)
    }


# For backward compatibility - some code might call verify_claim
def verify_claim(claim: str, evidence: str):
    """Alias for verify_with_evidence for backward compatibility"""
    return verify_with_evidence(claim, evidence)


# Example usage and testing
if __name__ == "__main__":
    test_claims = [
        "Global temperatures have increased significantly over the past century",
        "The Earth is flat",
        "Renewable energy is becoming more affordable",
    ]
    
    print("Testing NLI Model with Adjusted Thresholds\n")
    print("=" * 60)
    
    for claim in test_claims:
        classification, confidence = classify_claim(claim)
        print(f"\nClaim: {claim}")
        print(f"Classification: {classification} ({confidence:.2f}% confidence)")
        print("-" * 60)
