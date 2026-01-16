"""
Sarvam AI Backend for Tamil Text Processing
Uses HuggingFace API with sarvamai model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Sarvam AI Backend", description="Tamil Text Processing API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace Configuration from environment
HF_TOKEN = os.getenv("HF_TOKEN")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvamai/sarvam-2b-v0.5")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required. Set it in backend/.env file")

# Try new router endpoint first, fallback to direct model access
HF_API_URLS = [
    f"https://huggingface.co/api/models/{SARVAM_MODEL}",
    f"https://api-inference.huggingface.co/models/{SARVAM_MODEL}"
]

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Request Models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

class CorrectTamilRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = 'en'
    target_lang: str = 'ta'

print("=" * 80)
print("Sarvam AI Backend Server (FastAPI)")
print("=" * 80)
print(f"Model: {SARVAM_MODEL}")
print(f"Token: {HF_TOKEN[:20]}...")
print("=" * 80)


def call_sarvam_api(prompt, max_tokens=200, temperature=0.7):
    """Call Sarvam AI model via HuggingFace Inference API"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True
        }
    }
    
    try:
        # Note: HuggingFace Inference API for this model is currently unavailable (410 error)
        # This backend is configured and ready to use when the API becomes available
        # or when using local model deployment
        
        for api_url in HF_API_URLS:
            response = requests.post(api_url, headers=HF_HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return {"success": True, "text": result[0].get("generated_text", ""), "model": SARVAM_MODEL}
                elif isinstance(result, dict):
                    return {"success": True, "text": result.get("generated_text", ""), "model": SARVAM_MODEL}
                return {"success": False, "error": "Unexpected response"}
            
            elif response.status_code == 503:
                return {"success": False, "error": "Model loading (wait 20-30s)"}
            
            elif response.status_code != 410:  # Skip deprecated endpoints
                return {"success": False, "error": f"API Error {response.status_code}"}
        
        # If all endpoints fail
        return {
            "success": False,
            "error": "HuggingFace Inference API unavailable",
            "message": "The Sarvam model endpoint is currently deprecated. Backend is configured and ready for when the model becomes available or for local deployment."
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/")
def home():
    return {
        "service": "Sarvam AI Backend",
        "model": SARVAM_MODEL,
        "status": "running",
        "endpoints": {
            "POST /generate": "Generate text",
            "POST /correct-tamil": "Correct Tamil OCR text",
            "POST /translate": "Translate text",
            "GET /health": "Health check"
        }
    }


@app.get('/health')
def health():
    test = call_sarvam_api("Test", max_tokens=5)
    return {
        "status": "healthy" if test.get("success") else "degraded",
        "model": SARVAM_MODEL,
        "api_status": test.get("success", False)
    }


@app.post('/generate')
def generate(request: GenerateRequest):
    """Generate text using Sarvam AI"""
    try:
        result = call_sarvam_api(
            request.prompt,
            request.max_tokens,
            request.temperature
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "API error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/correct-tamil')
def correct_tamil(request: CorrectTamilRequest):
    """Correct Tamil OCR text"""
    try:
        prompt = f"""Correct Tamil text from OCR. Fix errors:

Original: {request.text}

Corrected:"""
        
        result = call_sarvam_api(prompt, max_tokens=300, temperature=0.5)
        
        if result.get("success"):
            return {
                "success": True,
                "original": request.text,
                "corrected": result.get("text", ""),
                "model": SARVAM_MODEL
            }
        
        raise HTTPException(status_code=500, detail=result.get("error", "API error"))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/translate')
def translate(request: TranslateRequest):
    """Translate text"""
    try:
        langs = {'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi'}
        
        prompt = f"Translate from {langs.get(request.source_lang, request.source_lang)} to {langs.get(request.target_lang, request.target_lang)}: {request.text}\n\nTranslation:"
        
        result = call_sarvam_api(prompt, max_tokens=300, temperature=0.3)
        
        if result.get("success"):
            return {
                "success": True,
                "original": request.text,
                "translated": result.get("text", "")
            }
        
        raise HTTPException(status_code=500, detail=result.get("error", "API error"))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n🚀 Starting Sarvam AI Backend Server (FastAPI)")
    print("=" * 80)
    print(f"Server: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print(f"Model: {SARVAM_MODEL}")
    print("=" * 80)
    print("\nEndpoints:")
    print("  POST /generate        - Generate text")
    print("  POST /correct-tamil   - Correct Tamil OCR text")
    print("  POST /translate       - Translate text")
    print("  GET  /health          - Health check")
    print("  GET  /docs            - Interactive API documentation")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
