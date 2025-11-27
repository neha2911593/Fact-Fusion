import streamlit as st
import requests
import time

# Page config
st.set_page_config(
    page_title="Fact Fusion - Climate Claims Verification",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    /* Fix for HTML rendering in expander */
    .streamlit-expanderContent {
        overflow-x: auto;
    }
    /* Ensure HTML content displays properly */
    iframe {
        border: none;
        width: 100%;
        min-height: 400px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌍 Fact Fusion</h1>
        <p>Automated Environment Misinformation Detection</p>
    </div>
""", unsafe_allow_html=True)

# Example claims
EXAMPLE_CLAIMS = [
    "Earth's temperature currently increased by 1.1 degree celsius",
    "Arctic sea ice is melting at an unprecedented rate",
    "Renewable energy cannot meet global energy demands",
    "Climate change is causing more frequent hurricanes",
    "The Earth's temperature has increased by more than 1 degree Celsius since 1880",
    "Fossil fuels are the primary driver of global warming",
    "Sea levels are rising due to melting ice caps",
    "Solar panels are more harmful to the environment than coal",
    "Global warming is caused by human activities"
]

# Two columns layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter a Climate or Weather Claim")
    
    if 'claim_input' not in st.session_state:
        st.session_state.claim_input = ""
    
    claim = st.text_area(
        "Claim to verify:",
        value=st.session_state.claim_input,
        height=100,
        placeholder="Enter a claim about climate or weather..."
    )
    
    verify_button = st.button("🔍 Verify Claim", type="primary")

with col2:
    st.subheader("💡 Example Claims")
    st.markdown("Click on any example to use it:")
    
    for i, example in enumerate(EXAMPLE_CLAIMS):
        if st.button(example, key=f"example_{i}", help="Click to use this claim"):
            st.session_state.claim_input = example
            st.rerun()

# Process verification
if verify_button and claim.strip():
    with st.spinner("🔄 Analyzing claim..."):
        start_time = time.time()
        
        try:
            # Call backend API
            response = requests.post(
                "http://localhost:8000/predict",
                json={"claim": claim},
                timeout=120
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Backend returns {"data": [result]}
                if "data" in response_data and len(response_data["data"]) > 0:
                    result = response_data["data"][0]
                else:
                    st.error("❌ Invalid response format from backend")
                    st.json(response_data)
                    st.stop()
                
                processing_time = time.time() - start_time
                
                st.markdown("---")
                
                # Cache indicator
                if result.get("from_cache"):
                    st.success(f"⚡ {result.get('cache_info', 'Retrieved from cache')}")
                    st.caption(f"Retrieval time: {result.get('cache_retrieval_time', 'N/A')}")
                
                st.header("📊 Results")
                
                # Your Claim
                st.subheader("📄 Your Claim")
                st.info(claim)
                
                # AI Classification
                st.subheader("🤖 AI Classification")
                prediction = result.get("Prediction", "Unknown")
                
                if "True" in prediction:
                    st.success(f"**{prediction}**")
                elif "False" in prediction:
                    st.error(f"**{prediction}**")
                else:
                    st.warning(f"**{prediction}**")
                
                st.caption("Model: RoBERTa-Large-MNLI (Hugging Face)")
                
                # Retrieved Evidence
                st.subheader("📚 Retrieved Evidence")
                evidence = result.get("Evidence", "No evidence found")
                
                with st.expander("View Evidence", expanded=True):
                    st.markdown(evidence)
                
                # NLI Verification
                st.subheader("🧠 Evidence-Based Verification (NLI)")
                nli_result = result.get("NLI Result", "Unknown")
                
                if "True" in nli_result:
                    st.success(f"**{nli_result}**")
                    st.success("✅ The claim is SUPPORTED by the retrieved evidence")
                elif "False" in nli_result:
                    st.error(f"**{nli_result}**")
                    st.error("❌ The claim is NOT SUPPORTED by the retrieved evidence")
                else:
                    st.warning(f"**{nli_result}**")
                    st.warning("⚠️ The evidence is INCONCLUSIVE regarding the claim")
                
                # Explanation - FIXED RENDERING
                explanation_html = result.get("Explanation (SHAP)", "")
                if explanation_html and "failed" not in explanation_html.lower():
                    st.subheader("🔍 Word Importance Analysis")
                    st.markdown("**Words colored by importance:**")
                    st.caption("Shows which words most influenced the classification")
                    
                    with st.expander("View Word Importance", expanded=True):
                        # Use components.html for better rendering
                        import streamlit.components.v1 as components
                        
                        # Wrap HTML for proper rendering
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <style>
                                body {{
                                    margin: 0;
                                    padding: 10px;
                                    font-family: Arial, sans-serif;
                                }}
                            </style>
                        </head>
                        <body>
                            {explanation_html}
                        </body>
                        </html>
                        """
                        
                        components.html(full_html, height=500, scrolling=True)
                    
                    st.caption("Method: SHAP/Attention - Shows which words most influenced the classification")
                else:
                    st.info("ℹ️ Explanation not available for this claim")
                
                # Performance Metrics
                with st.expander("⚡ Performance Metrics"):
                    perf = result.get("Performance", {})
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        total = perf.get("total_time", f"{processing_time:.2f}s")
                        st.metric("Total Time", total)
                    with col_m2:
                        st.metric("Evidence Time", perf.get("evidence_time", "N/A"))
                    with col_m3:
                        st.metric("NLI Time", perf.get("nli_time", "N/A"))
                    
                    col_m4, col_m5, col_m6 = st.columns(3)
                    with col_m4:
                        st.metric("Classification", perf.get("classifier_time", "N/A"))
                    with col_m5:
                        st.metric("Explanation", perf.get("explanation_time", "N/A"))
                    with col_m6:
                        if result.get("from_cache"):
                            st.caption(f"⚡ From cache")
                        else:
                            st.caption(f"⏱️ Fresh processing")
                
                st.caption("Powered by RoBERTa-Large-MNLI • Evidence from NASA, NOAA, Wikipedia & more")
                
            else:
                st.error(f"❌ Backend Error {response.status_code}")
                st.code(response.text)
                st.info("**Troubleshooting:**\n- Make sure backend is running: `python backend_improved.py`\n- Check terminal for errors")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The backend might be processing. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 **Cannot connect to backend**")
            st.error("Please start the backend:")
            st.code("python backend_improved.py")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("Show full error"):
                import traceback
                st.code(traceback.format_exc())

elif verify_button:
    st.warning("⚠️ Please enter a claim to verify")

# Sidebar
with st.sidebar:
    st.header("ℹ️ How it works")
    
    st.markdown("""
    ### 🤖 AI Classification
    Uses **RoBERTa-Large-MNLI** model to classify claims
    
    ### 📚 Multi-Source Evidence
    Retrieves data from:
    - NASA Climate Data
    - NOAA NCEI
    - IPCC Reports
    - Wikipedia
    - DuckDuckGo
    
    ### 🔍 Explainability
    **SHAP/Attention** word importance analysis
    
    ### ✅ Verification Process
    1. Enter your claim
    2. AI classifies the claim
    3. Evidence is retrieved
    4. Cross-verification with NLI
    5. Results with explanations
    
    ### ⚡ Caching
    Previously searched claims are cached for 24 hours
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tips")
    st.markdown("""
    - Be specific in your claims
    - Use complete sentences
    - Avoid ambiguous language
    - Include measurable statements
    """)
    
    st.markdown("---")
    
    # Backend status
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Backend running")
            health_data = health.json()
            
            # Get cache stats
            try:
                stats = requests.get("http://localhost:8000/cache-stats", timeout=2)
                if stats.status_code == 200:
                    stats_data = stats.json()
                    st.caption(f"📊 Cache: {stats_data['statistics']['currently_cached']} claims")
                    st.caption(f"📈 Hit rate: {stats_data['statistics']['hit_rate']}")
            except:
                pass
        else:
            st.error("❌ Backend unhealthy")
    except:
        st.error("❌ Backend offline")
        st.caption("Start with:")
        st.code("python backend_improved.py", language="bash")
