from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load RoBERTa-MNLI for evidence-based verification
mnli_model_name = "roberta-large-mnli"
mnli_tokenizer = AutoTokenizer.from_pretrained(mnli_model_name)
mnli_model = AutoModelForSequenceClassification.from_pretrained(mnli_model_name)

# Load Zero-Shot model for claim classification (without evidence)
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

DEBUG = False

def _debug_print(message):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[NLI DEBUG] {message}")


def classify_claim_simple(claim: str):
    """
    Simplified version - just True/False categories (no Uncertain option)
    Uses NLI-style reasoning with scientific context
    
    Args:
        claim: The claim text to classify
        
    Returns:
        tuple: (classification_label, confidence_score)
    """
    # Create premise with scientific/factual context
    premise_true = "Scientific evidence and established facts demonstrate that"
    premise_false = "Scientific evidence and established facts show the opposite of"
    
    # Use MNLI model directly for better accuracy
    inputs_true = mnli_tokenizer(
        premise_true,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    inputs_false = mnli_tokenizer(
        premise_false,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs_true = mnli_model(**inputs_true)
        outputs_false = mnli_model(**inputs_false)
    
    probs_true = torch.softmax(outputs_true.logits, dim=1)
    probs_false = torch.softmax(outputs_false.logits, dim=1)
    
    # Get entailment scores (index 2)
    entailment_true = probs_true[0][2].item()
    entailment_false = probs_false[0][2].item()
    
    # Also check contradiction scores (index 0) 
    contradiction_true = probs_true[0][0].item()
    contradiction_false = probs_false[0][0].item()
    
    _debug_print(f"True entailment: {entailment_true:.4f}, False entailment: {entailment_false:.4f}")
    _debug_print(f"True contradiction: {contradiction_true:.4f}, False contradiction: {contradiction_false:.4f}")
    
    # Combined scoring - true if scientific evidence entails it
    # False if scientific evidence contradicts it or entails its opposite
    score_for_true = entailment_true + contradiction_false
    score_for_false = entailment_false + contradiction_true
    
    if score_for_true > score_for_false:
        classification = "True"
        confidence = (score_for_true / (score_for_true + score_for_false)) * 100
    else:
        classification = "False"
        confidence = (score_for_false / (score_for_true + score_for_false)) * 100
    
    return classification, confidence


def classify_claim(claim: str):
    """
    Classify a claim as True, False, or Uncertain using zero-shot classification
    
    Args:
        claim: The claim text to classify
        
    Returns:
        tuple: (classification_label, confidence_score)
    """
    _debug_print(f"classify_claim (zero-shot) called with: {claim[:100]}")
    
    candidate_labels = [
        "This statement is factually correct and scientifically accurate",
        "This statement is factually incorrect or false",
        "This statement is uncertain or unverifiable"
    ]
    
    result = zero_shot_classifier(
        claim,
        candidate_labels,
        multi_label=False
    )
    
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    _debug_print(f"Zero-shot scores: {list(zip(result['labels'], result['scores']))}")
    
    if "correct" in top_label.lower():
        classification = "True"
    elif "incorrect" in top_label.lower() or "false" in top_label.lower():
        classification = "False"
    else:
        classification = "Uncertain"
    
    if top_score < 0.40 and classification != "Uncertain":
        classification = "Uncertain"
    
    confidence = top_score * 100
    
    _debug_print(f"classify_claim result: {classification} ({confidence:.2f}%)")
    return classification, confidence


def verify_with_evidence(claim: str, evidence: str):
    """
    Verify a claim against retrieved evidence using RoBERTa-MNLI
    
    Args:
        claim: The claim to verify
        evidence: The evidence text retrieved from sources
        
    Returns:
        tuple: (verification_label, confidence_score as percentage)
    """
    _debug_print(f"verify_with_evidence (RoBERTa-MNLI) called")
    _debug_print(f"Claim: {claim[:100]}")
    _debug_print(f"Evidence: {evidence[:150]}...")
    
    # Truncate evidence if too long
    evidence_words = evidence.split()
    if len(evidence_words) > 500:
        evidence = " ".join(evidence_words[:500]) + "..."
    
    # Tokenize the claim-evidence pair
    inputs = mnli_tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = mnli_model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(outputs.logits, dim=1)
    
    # RoBERTa-MNLI outputs: [contradiction, neutral, entailment]
    contradiction_score = probs[0][0].item()
    neutral_score = probs[0][1].item()
    entailment_score = probs[0][2].item()
    
    _debug_print(f"MNLI scores - E: {entailment_score:.4f}, C: {contradiction_score:.4f}, N: {neutral_score:.4f}")
    
    # Simply pick the highest score
    if entailment_score >= contradiction_score and entailment_score >= neutral_score:
        result = ("True", entailment_score * 100)
    elif contradiction_score > entailment_score and contradiction_score >= neutral_score:
        result = ("False", contradiction_score * 100)
    else:
        result = ("Uncertain", neutral_score * 100)
    
    _debug_print(f"verify_with_evidence result: {result[0]} ({result[1]:.2f}%)")
    return result


def verify_claim(claim: str, evidence: str):
    """Alias for verify_with_evidence for backward compatibility"""
    return verify_with_evidence(claim, evidence)


def get_nli_scores(claim: str, evidence: str):
    """
    Get detailed NLI scores for debugging and analysis
    
    Args:
        claim: The claim to verify
        evidence: The evidence text
        
    Returns:
        dict: Dictionary containing all three scores
    """
    inputs = mnli_tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = mnli_model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    
    scores = {
        "contradiction": round(probs[0][0].item(), 4),
        "neutral": round(probs[0][1].item(), 4),
        "entailment": round(probs[0][2].item(), 4)
    }
    
    _debug_print(f"get_nli_scores: {scores}")
    return scores


# Example usage and testing
if __name__ == "__main__":
    print("Testing NLI Model Functions")
    print("="*80)
    
    test_claim = "Climate change is caused by human activities"
    test_evidence = "According to NASA and IPCC reports, human activities are the dominant cause of observed climate change since the mid-20th century."
    
    print("\n1. Testing classify_claim_simple (True/False only):")
    label, conf = classify_claim_simple(test_claim)
    print(f"   Result: {label} ({conf:.2f}%)")
    
    print("\n2. Testing classify_claim (with Uncertain option):")
    label, conf = classify_claim(test_claim)
    print(f"   Result: {label} ({conf:.2f}%)")
    
    print("\n3. Testing verify_with_evidence:")
    label, conf = verify_with_evidence(test_claim, test_evidence)
    print(f"   Result: {label} ({conf:.2f}%)")
    
    print("\n" + "="*80)
