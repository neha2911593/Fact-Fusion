import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional

# FIXED IMPORTS
from classifier_model import classify_claim
from nli_model import verify_with_evidence
from explain_shap import explain_claim  
from improved_evidence_retriever import get_evidence

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FactFusion API - Enhanced with Caching")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CACHING SYSTEM
# ============================================================================
CACHE = {}
CACHE_EXPIRY_HOURS = 24
CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "total_requests": 0
}

def get_cache_key(claim: str) -> str:
    """Generate a unique cache key for a claim"""
    normalized_claim = claim.strip().lower()
    return hashlib.md5(normalized_claim.encode()).hexdigest()

def get_cached_result(claim: str) -> Optional[dict]:
    """Retrieve cached result if available and not expired"""
    cache_key = get_cache_key(claim)
    
    if cache_key in CACHE:
        cache_entry = CACHE[cache_key]
        
        if datetime.now() - cache_entry["timestamp"] < timedelta(hours=CACHE_EXPIRY_HOURS):
            CACHE_STATS["hits"] += 1
            logger.info(f"✅ CACHE HIT: {claim[:50]}...")
            return cache_entry["result"]
        else:
            del CACHE[cache_key]
            logger.info(f"⏰ CACHE EXPIRED: {claim[:50]}...")
    
    CACHE_STATS["misses"] += 1
    logger.info(f"❌ CACHE MISS: {claim[:50]}...")
    return None

def cache_result(claim: str, result: dict):
    """Store result in cache"""
    cache_key = get_cache_key(claim)
    CACHE[cache_key] = {
        "result": result,
        "timestamp": datetime.now(),
        "claim": claim
    }
    logger.info(f"💾 CACHED: {claim[:50]}...")

# ============================================================================
# REQUEST MODELS
# ============================================================================
class ClaimRequest(BaseModel):
    claim: str | None = None
    data: list[str] | None = None

class VerifyClaimRequest(BaseModel):
    claim: str

SERPER_API_KEY = os.getenv("SERPER_API_KEY", None)

if SERPER_API_KEY:
    logger.info("✓ Serper API key found - enhanced search enabled")
else:
    logger.info("✗ No Serper API key - using Wikipedia + DuckDuckGo only")

# ============================================================================
# SHARED PROCESSING FUNCTION
# ============================================================================
def process_claim(claim: str):
    """Shared function to process claims - used by both endpoints"""
    CACHE_STATS["total_requests"] += 1
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"Request #{CACHE_STATS['total_requests']}")
    logger.info(f"Claim: {claim}")

    # CHECK CACHE
    cached_result = get_cached_result(claim)
    if cached_result:
        cache_time = time.time() - start_time
        cached_result["from_cache"] = True
        cached_result["cache_retrieval_time"] = f"{cache_time:.2f}s"
        cached_result["cache_info"] = "⚡ Retrieved from cache"
        
        logger.info(f"✓ Cached response: {cache_time:.2f}s")
        logger.info("=" * 60)
        return cached_result

    # STEP 1: Classify
    logger.info("[1/4] Classifying...")
    classifier_start = time.time()
    try:
        label, confidence = classify_claim(claim)
        classifier_time = time.time() - classifier_start
        logger.info(f"✓ Classification: {label} ({confidence:.2f}%) - {classifier_time:.2f}s")
    except Exception as e:
        logger.exception("Classifier error")
        label, confidence = "Uncertain", 0.0
        classifier_time = time.time() - classifier_start

    # STEP 2: Evidence
    logger.info("[2/4] Fetching evidence...")
    evidence_start = time.time()
    try:
        evidence = get_evidence(claim, serper_api_key=SERPER_API_KEY)
        evidence_time = time.time() - evidence_start
        
        if "No relevant evidence" in evidence:
            logger.warning(f"✗ No evidence - {evidence_time:.2f}s")
        else:
            logger.info(f"✓ Evidence: {len(evidence)} chars - {evidence_time:.2f}s")
            
    except Exception as e:
        logger.exception("Evidence error")
        evidence = "Evidence retrieval failed."
        evidence_time = time.time() - evidence_start

    # STEP 3: NLI
    logger.info("[3/4] NLI verification...")
    nli_start = time.time()
    try:
        if "No relevant evidence" not in evidence and "failed" not in evidence:
            nli_label, nli_conf = verify_with_evidence(claim, evidence)
            nli_time = time.time() - nli_start
            logger.info(f"✓ NLI: {nli_label} ({nli_conf:.2f}%) - {nli_time:.2f}s")
        else:
            nli_label, nli_conf = "Uncertain", 0.0
            nli_time = time.time() - nli_start
            logger.warning(f"✗ NLI skipped - {nli_time:.2f}s")
            
    except Exception as e:
        logger.exception("NLI error")
        nli_label, nli_conf = "Uncertain", 0.0
        nli_time = time.time() - nli_start

    # STEP 4: Explanation
    logger.info("[4/4] Generating explanation...")
    explanation_start = time.time()
    try:
        explanation_html = explain_claim(claim, method="simple")
        explanation_time = time.time() - explanation_start
        logger.info(f"✓ Explanation - {explanation_time:.2f}s")
        
    except Exception as e:
        logger.exception("Explanation error")
        explanation_html = f"<p style='color:red;'>Explanation failed: {str(e)}</p>"
        explanation_time = time.time() - explanation_start

    # Build Response
    total_time = time.time() - start_time
    
    result = {
        "Claim": claim,
        "Prediction": f"{label} ({confidence:.2f}%)",
        "Evidence": evidence,
        "NLI Result": f"{nli_label} ({nli_conf:.2f}%)",
        "Explanation (SHAP)": explanation_html,
        "Performance": {
            "total_time": f"{total_time:.2f}s",
            "classifier_time": f"{classifier_time:.2f}s",
            "evidence_time": f"{evidence_time:.2f}s",
            "nli_time": f"{nli_time:.2f}s",
            "explanation_time": f"{explanation_time:.2f}s"
        },
        "from_cache": False,
        # Additional fields for app.py compatibility
        "classification": label,
        "classification_confidence": confidence / 100,  # Convert back to 0-1 range
        "evidence": evidence,
        "verification": nli_label,
        "verification_confidence": nli_conf / 100,
        "explanation": explanation_html,
        "metrics": {
            "total_time": total_time,
            "classifier_time": classifier_time,
            "evidence_time": evidence_time,
            "nli_time": nli_time,
            "explanation_time": explanation_time
        }
    }

    # Cache it
    cache_result(claim, result)

    logger.info(f"✓ Total: {total_time:.2f}s")
    logger.info("=" * 60)
    
    return result

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    cache_hit_rate = (CACHE_STATS["hits"] / CACHE_STATS["total_requests"] * 100) if CACHE_STATS["total_requests"] > 0 else 0
    
    return {
        "message": "Fact Fusion API - Enhanced with Caching",
        "version": "2.1",
        "endpoints": ["/predict", "/verify-claim"],
        "features": [
            "✓ Improved evidence retrieval",
            "✓ Fast SHAP explanations",
            "✓ Robust NLI verification",
            "✓ 24h caching system"
        ],
        "cache": {
            "cached_claims": len(CACHE),
            "hit_rate": f"{cache_hit_rate:.1f}%",
            "total_requests": CACHE_STATS["total_requests"]
        }
    }


@app.post("/predict")
async def predict(body: ClaimRequest):
    """Original endpoint for backward compatibility"""
    # Extract claim
    if body.claim:
        claim = body.claim.strip()
    elif body.data and len(body.data) > 0:
        claim = body.data[0].strip()
    else:
        return {"error": "Invalid input format. Expected {'claim': 'text'}"}

    if not claim:
        return {"error": "Empty claim provided"}

    result = process_claim(claim)
    return {"data": [result]}


@app.post("/verify-claim")
async def verify_claim_endpoint(body: VerifyClaimRequest):
    """New endpoint for Streamlit app"""
    claim = body.claim.strip()
    
    if not claim:
        return {"error": "Empty claim provided"}
    
    result = process_claim(claim)
    return result  # Return result directly (not wrapped in {"data": []})


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "serper_enabled": SERPER_API_KEY is not None,
        "cache_enabled": True,
        "cached_claims": len(CACHE)
    }


@app.get("/cache-stats")
async def cache_stats():
    """Cache statistics"""
    expired_count = 0
    for cache_key, cache_entry in list(CACHE.items()):
        if datetime.now() - cache_entry["timestamp"] > timedelta(hours=CACHE_EXPIRY_HOURS):
            del CACHE[cache_key]
            expired_count += 1
    
    total = CACHE_STATS["total_requests"]
    hit_rate = (CACHE_STATS["hits"] / total * 100) if total > 0 else 0
    
    cache_contents = []
    for cache_entry in list(CACHE.values())[:10]:
        age = (datetime.now() - cache_entry["timestamp"]).total_seconds() / 3600
        cache_contents.append({
            "claim": cache_entry["claim"][:60] + "...",
            "age_hours": round(age, 2),
            "expires_in": round(CACHE_EXPIRY_HOURS - age, 2)
        })
    
    return {
        "statistics": {
            "total_requests": total,
            "cache_hits": CACHE_STATS["hits"],
            "cache_misses": CACHE_STATS["misses"],
            "hit_rate": f"{hit_rate:.2f}%",
            "currently_cached": len(CACHE),
            "expired_removed": expired_count
        },
        "sample_cached_claims": cache_contents
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear cache"""
    size = len(CACHE)
    CACHE.clear()
    logger.info(f"🗑️ Cache cleared: {size} entries")
    
    return {
        "status": "success",
        "entries_removed": size
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Fact Fusion Backend")
    logger.info("Features:")
    logger.info("  - Multi-source evidence retrieval")
    logger.info("  - Fast SHAP explanations")
    logger.info("  - Comprehensive logging")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
