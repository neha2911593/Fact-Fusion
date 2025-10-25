import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fact Fusion: Climate & Weather Edu-Bot", layout="centered")
st.title("🌿 Fact Fusion: Climate & Weather Edu-Bot")
st.write("Ask me about climate, weather, or environment facts! I’ll check if it’s true and show you evidence.")

user_input = st.text_input("Enter your claim or question:")

if st.button("Check Fact"):
    if user_input:
        with st.spinner("Analyzing your claim... please wait ⏳"):
            try:
                # ✅ FIX: send 'claim' field instead of 'data'
                response = requests.post(
                    BACKEND_URL,
                    json={"claim": user_input},  # ✅ correct field
                    timeout=600
                )

                if response.status_code == 200:
                    result = response.json()

                    # Backend sends {"data": [ { ..results.. } ]}
                    if "data" in result and len(result["data"]) > 0:
                        output = result["data"][0]

                        st.subheader("Prediction:")
                        st.write(output.get("Prediction", "N/A"))

                        st.subheader("Evidence from Wikipedia/NASA:")
                        st.write(output.get("Evidence", "N/A"))

                        st.subheader("NLI Verification:")
                        st.write(output.get("NLI Result", "N/A"))

                        st.subheader("Explainability (LIME):")
                        st.components.v1.html(output.get("Explanation (LIME)", ""), height=400, scrolling=True)
                    else:
                        st.error("Unexpected backend response format.")
                else:
                    st.error(f"Backend returned HTTP {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to backend. Please ensure `backend.py` is running first.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
