import streamlit as st
import requests

# Make sure this matches your backend port
BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fact Fusion: Climate & Fact Checker", layout="centered")

st.title("🌍 Fact Fusion: Climate, Weather & Environment Fact Checker")
st.write("Enter any claim about climate, weather, or the environment. I'll verify it using:")
st.markdown("""
- 🌤 **Roberta NLI Model (Fact Classification)**  
- 📚 **Wikipedia + NASA + DuckDuckGo Evidence Retrieval**  
- 🧠 **LIME Explainability**  
""")

user_input = st.text_input("Enter your claim:")

if st.button("Verify Claim"):
    if user_input.strip():
        with st.spinner("Processing your claim... 🔍"):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"claim": user_input},
                    timeout=1000
                )

                if response.status_code == 200:
                    result = response.json()

                    # ----------------------------
                    # Expected Format:
                    # { "data": [{ "Prediction": "...", "Evidence": "...", ... }] }
                    # ----------------------------
                    if "data" in result and len(result["data"]) > 0:
                        output = result["data"][0]

                        # PREDICTION
                        st.subheader("📌 Claim Classification (Roberta-NLI)")
                        st.success(output.get("Prediction", "Unavailable"))

                        # EVIDENCE
                        st.subheader("📚 Retrieved Evidence")
                        st.write(output.get("Evidence", "No evidence found."))

                        # NLI VERIFICATION
                        st.subheader("🧠 NLI Verification Result")
                        st.info(output.get("NLI Result", "Unavailable"))

                        # LIME EXPLANATION
                        st.subheader("🔍 Explainability (LIME)")
                        lime_html = output.get("Explanation (LIME)", "")
                        st.components.v1.html(lime_html, height=450, scrolling=True)

                    else:
                        st.error("❌ Invalid backend response format.")

                else:
                    st.error(f"⚠️ Backend Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚫 Backend not reachable. Start backend.py first!")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please enter a claim before clicking the button.")

