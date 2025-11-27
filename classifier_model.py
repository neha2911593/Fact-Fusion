from nli_model import classify_claim_simple as nli_classify_claim
from nli_model import verify_with_evidence as nli_verify_evidence

def classify_claim(claim):
    """
    Classify a claim using the proper NLI model function (zero-shot)
    
    Args:
        claim: The claim text to classify
        
    Returns:
        tuple: (label, confidence)
    """
    label, confidence = nli_classify_claim(claim)
    return label, round(confidence, 2)


def verify_claim_with_evidence(claim, evidence):
    """
    Verify a claim against evidence using NLI
    
    Args:
        claim: The claim to verify
        evidence: The evidence text from search results
        
    Returns:
        tuple: (label, confidence)
    """
    label, confidence = nli_verify_evidence(claim, evidence)
    return label, round(confidence, 2)
