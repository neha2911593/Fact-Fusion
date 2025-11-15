import gradio as gr
from fastapi import FastAPI, Request
from classifier_model import classify_claim       # Updated classifier (roberta-large-nli)
from wiki_retriever import get_evidence           # Updated evidence retriever (Wikipedia REST + NASA + DuckDuckGo etc.)
from nli_model import verify_claim                # Updated NLI verification
from explain_lime import explain_with_lime        # Updated LIME explainability

# ------------------------ FACT FUSION PIPELINE ------------------------
def fact_fusion_pipeline(claim):
    # 1. Classify claim (True/False)
    label, confidence = classify_claim(claim)

    # 2. Retrieve evidence from APIs (Wikipedia REST + NASA + DuckDuckGo etc.)
    evidence = get_evidence(claim)

    # 3. Verify claim via NLI (entailment / contradiction / neutral)
    nli_label, nli_conf = verify_claim(claim, evidence)

    # 4. Explain with LIME (HTML heatmap)
    explanation_html = explain_with_lime(claim)

    return {
        "Claim": claim,
        "Prediction": f"{label} ({confidence:.2f}%)",
        "Evidence": evidence,
        "NLI Result": f"{nli_label} ({nli_conf:.2f}%)",
        "Explanation (LIME)": explanation_html
    }

# ------------------------ FASTAPI BACKEND -----------------------------
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    """
    Supports formats:
    1. {"claim": "text"}
    2. {"data": ["text"]}
    """
    try:
        body = await request.json()

        # Case 1: Streamlit frontend format
        if "claim" in body:
            claim = body["claim"]

        # Case 2: Gradio / Batch format
        elif "data" in body and isinstance(body["data"], list) and len(body["data"]) > 0:
            claim = body["data"][0]

        else:
            return {"error": "Invalid input format. Expected: {'claim': 'text'}"}

        result = fact_fusion_pipeline(claim)
        return {"data": [result]}

    except Exception as e:
        return {"error": str(e)}

# ------------------------ OPTIONAL GRADIO UI --------------------------
"""
with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Fact Fusion: Climate & Weather Edu-Bot")
    gr.Markdown("Detect misinformation, fetch evidence, and explain decisions.")

    with gr.Row():
        claim_input = gr.Textbox(label="Enter a Climate Claim")
        run_btn = gr.Button("Check Claim")

    pred_output = gr.Textbox(label="Prediction")
    evidence_output = gr.Textbox(label="Evidence")
    nli_output = gr.Textbox(label="NLI Verification")
    lime_output = gr.HTML(label="Explainability (LIME)")

    def run_pipeline_text(claim):
        result = fact_fusion_pipeline(claim)
        return (result["Prediction"], result["Evidence"],
                result["NLI Result"], result["Explanation (LIME)"])

    run_btn.click(run_pipeline_text, inputs=claim_input,
                  outputs=[pred_output, evidence_output, nli_output, lime_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
"""

# ------------------------ FASTAPI SERVER RUNNER -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
