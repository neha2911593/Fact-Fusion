from lime.lime_text import LimeTextExplainer
from models import nli_model     # shared global model
import numpy as np

# LIME class names in 3-class order
class_names = ["False", "Uncertain", "True"]
explainer = LimeTextExplainer(class_names=class_names)

def explain_with_lime(text):
    def predict_proba(texts):
        results = nli_model(texts)

        probs = []
        for r in results:
            score = r["score"]
            label = r["label"]

            # Convert NLI label → probability vector
            if label == "CONTRADICTION":
                probs.append([score, (1-score)/2, (1-score)/2])

            elif label == "NEUTRAL":
                probs.append([(1-score)/2, score, (1-score)/2])

            else:  # ENTAILMENT
                probs.append([(1-score)/2, (1-score)/2, score])

        return np.array(probs)

    try:
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        return exp.as_html()

    except Exception as e:
        return f"<p style='color:red;'>LIME explanation failed: {str(e)}</p>"
