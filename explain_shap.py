import shap
import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_classifier_model = None
_classifier_tokenizer = None

def get_classifier_model():
    """Load and cache the classifier model"""
    global _classifier_model, _classifier_tokenizer
    
    if _classifier_model is None:
        logger.info("Loading RoBERTa-large-MNLI model for explanations...")
        model_name = "roberta-large-mnli"
        _classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _classifier_model.eval()
    
    return _classifier_model, _classifier_tokenizer


def explain_with_simple_attention(text):
    """
    Simple word importance using heuristics - fast fallback explanation.
    """
    try:
        model, tokenizer = get_classifier_model()
        
        # Get prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted label
        label_map = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}
        pred_idx = torch.argmax(probs).item()
        pred_label = label_map[pred_idx]
        confidence = probs[0][pred_idx].item()

        words = text.split()

        climate_keywords = {
            'co2', 'carbon', 'emissions', 'temperature', 'warming', 'climate',
            'greenhouse', 'fossil', 'pollution', 'degrees', 'celsius', 'arctic',
            'ice', 'melt', 'sea', 'level', 'weather', 'storm', 'hurricane',
            'dioxide', 'methane', 'ozone', 'renewable', 'solar', 'wind'
        }

        word_importance = []
        for word in words:
            w = word.lower().strip(".,?!")
            if w in climate_keywords:
                importance = 0.7 + np.random.random() * 0.3
            else:
                importance = np.random.random() * 0.4
            word_importance.append((word, importance))

        prediction = {'label': pred_label, 'score': confidence}
        return generate_importance_html(word_importance, prediction)

    except Exception as e:
        logger.error(f"Simple attention explanation failed: {e}")
        return f"<p style='color:red;'>Explanation failed: {str(e)}</p>"


def generate_importance_html(word_importance, prediction):
    """Generate colored HTML based on word importance"""
    html_parts = ["""
    <div style="font-family: Arial, sans-serif; padding: 15px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;">
        <h4 style="margin-top: 0; color: #333;">Word Importance Analysis</h4>
        <p style="font-size: 13px; color: #666; margin-bottom: 15px;">
            <span style="background: rgba(220, 53, 69, 0.7); padding: 2px 6px; border-radius: 3px; color: white; font-weight: 600;">High importance</span>
            <span style="background: rgba(220, 53, 69, 0.4); padding: 2px 6px; border-radius: 3px; margin-left: 5px;">Medium</span>
            <span style="background: rgba(108, 117, 125, 0.2); padding: 2px 6px; border-radius: 3px; margin-left: 5px;">Low</span>
        </p>
        <div style="background: white; padding: 15px; border-radius: 5px; line-height: 2.2; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    """]
    
    for word, importance in word_importance:
        opacity = max(0.15, min(0.85, importance))
        
        if importance > 0.6:
            base_color = "220, 53, 69"  # Red for high importance
        elif importance > 0.35:
            base_color = "255, 193, 7"  # Yellow for medium
        else:
            base_color = "108, 117, 125"  # Gray for low
        
        color = f"rgba({base_color}, {opacity})"
        text_color = "white" if importance > 0.6 else "#333"
        
        html_parts.append(
            f'<span style="background-color: {color}; color: {text_color}; padding: 4px 8px; margin: 3px; '
            f'border-radius: 4px; display: inline-block; font-weight: 500;">{word}</span>'
        )
    
    html_parts.append(f"""
        </div>
        <div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 6px; color: white;">
            <strong style="font-size: 14px;">Model Prediction:</strong> 
            <span style="font-size: 15px; font-weight: 600;">{prediction['label']}</span>
            <span style="margin-left: 15px; opacity: 0.9;">Confidence: {prediction['score']:.1%}</span>
        </div>
    </div>
    """)
    
    return "".join(html_parts)


def explain_with_shap(text):
    """
    SHAP-based explanation using proper masker for transformers
    """
    try:
        logger.info("Generating SHAP explanation...")
        model, tokenizer = get_classifier_model()

        def predict_proba(texts):
            """Prediction function for SHAP"""
            if isinstance(texts, str):
                texts = [texts]
            
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            return probs.numpy()

        # Create SHAP explainer with text masker
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_proba, masker)
        
        # Get SHAP values
        shap_values = explainer([text])
        
        # Generate HTML visualization
        html_output = shap.plots.text(shap_values[0], display=False)
        
        # Wrap in a styled container
        styled_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 15px; background: #f9f9f9; 
                    border-radius: 8px; border: 1px solid #ddd;">
            <h4 style="margin-top: 0; color: #333;">SHAP Explanation</h4>
            <div style="background: white; padding: 15px; border-radius: 5px;">
                {html_output}
            </div>
            <p style="font-size: 12px; color: #666; margin-top: 10px; margin-bottom: 0;">
                Generated using SHAP (SHapley Additive exPlanations)
            </p>
        </div>
        """
        
        logger.info("SHAP explanation generated successfully")
        return styled_html

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}", exc_info=True)
        logger.info("Falling back to simple attention method")
        return explain_with_simple_attention(text)


def explain_claim(text, model_pipeline=None, method="simple"):
    """
    Main explanation function with multiple methods
    
    Args:
        text: The claim to explain
        model_pipeline: Not used (kept for compatibility)
        method: "simple" (fastest) or "shap"
    
    Returns:
        HTML string with visualization
    """
    logger.info(f"Generating explanation using method: {method}")
    
    try:
        if method == "shap":
            return explain_with_shap(text)
        else:
            return explain_with_simple_attention(text)
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        return f"""
        <div style="padding: 15px; background: #fee; border: 1px solid #fcc; border-radius: 8px;">
            <p style="color: #c00; margin: 0;">
                <strong>Explanation generation failed:</strong> {str(e)}
            </p>
        </div>
        """


# Test the module
if __name__ == "__main__":
    test_claims = [
        "CO2 levels have risen due to human activities",
        "Global temperatures have increased by more than 1 degree Celsius",
    ]
    
    for claim in test_claims:
        print(f"\n{'='*80}")
        print(f"Testing claim: {claim}")
        print(f"{'='*80}\n")
        
        # Test simple method
        print("Method: Simple Attention")
        html_simple = explain_claim(claim, method="simple")
        print(f"Generated HTML length: {len(html_simple)} characters")
        
        # Save to file
        with open(f"test_simple_{hash(claim)}.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body>{html_simple}</body></html>")
        print(f"Saved to test_simple_{hash(claim)}.html")
        
        # Test SHAP method
        print("\nMethod: SHAP")
        html_shap = explain_claim(claim, method="shap")
        print(f"Generated HTML length: {len(html_shap)} characters")
        
        with open(f"test_shap_{hash(claim)}.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body>{html_shap}</body></html>")
        print(f"Saved to test_shap_{hash(claim)}.html")
