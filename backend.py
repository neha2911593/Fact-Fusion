import gradio as gr
from fastapi import FastAPI, Request
from model.classifier_model import classify_claim
from retriever.wiki_retriever import get_evidence
from model.nli_model import verify_claim
from model.explain_lime import explain_with_lime

# ---------------- PIPELINE ------------------------
def fact_fusion_pipeline(claim):
    label, confidence = classify_claim(claim)
    evidence = get_evidence(claim)
    nli_result = verify_claim(claim, evidence)
    explanation_html = explain_with_lime(claim)

    return {
        "Claim": claim,
        "Prediction": f"{label} ({confidence:.2f}%)",
        "Evidence": evidence,
        "NLI Result": nli_result,
        "Explanation (LIME)": explanation_html
    }

# ---------------- FASTAPI APP ----------------
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    """
    This endpoint supports both formats:
    1. {"claim": "text"}
    2. {"data": ["text"]}
    """
    try:
        body = await request.json()

        # Accept both payload formats
        if "claim" in body:
            claim = body["claim"]
        elif "data" in body and isinstance(body["data"], list) and len(body["data"]) > 0:
            claim = body["data"][0]
        else:
            return {"error": "Invalid request format"}

        result = fact_fusion_pipeline(claim)
        return {"data": [result]}
    except Exception as e:
        return {"error": str(e)}

# ---------------- GRADIO UI ----------------

# ---------------- GRADIO UI for Gradio 5+ ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Fact Fusion: Climate & Weather Edu-Bot")
    gr.Markdown("Detect misinformation, retrieve evidence, and explain AI reasoning.")
    
    with gr.Row():
        claim_input = gr.Textbox(label="Enter a claim about Climate or Weather")
        submit_btn = gr.Button("Check Claim")
    
    with gr.Row():
        pred_output = gr.Textbox(label="Prediction")
        evidence_output = gr.Textbox(label="Evidence from Wikipedia/NASA")
        nli_output = gr.Textbox(label="NLI Verification")
        lime_output = gr.HTML(label="Explainability with LIME")
    
    def run_pipeline(claim):
        result = fact_fusion_pipeline(claim)
        return result["Prediction"], result["Evidence"], result["NLI Result"], result["Explanation (LIME)"]
    
    submit_btn.click(run_pipeline, inputs=claim_input, outputs=[pred_output, evidence_output, nli_output, lime_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
