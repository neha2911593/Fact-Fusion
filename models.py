from transformers import pipeline

# Load only ONCE (shared model)
nli_model = pipeline(
    "text-classification",
    model_name = "facebook/bart-large-mnli"

)
