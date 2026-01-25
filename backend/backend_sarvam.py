"""
Sarvam AI Backend for Tamil Text Processing
Uses local Sarvam model deployment via transformers
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Model Configuration
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvamai/sarvam-2b-v0.5")
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("Sarvam AI Backend Server (FastAPI) - Local Model")
print("=" * 80)
print(f"Model: {SARVAM_MODEL}")
print(f"Device: {device}")
print("=" * 80)
print("Loading Sarvam model locally...")

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        SARVAM_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    print(f"[OK] Sarvam model loaded successfully on {device}!")
    MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Failed to load Sarvam model: {e}")
    print("[INFO] Model will be downloaded on first use (this may take time)")
    tokenizer = None
    model = None
    MODEL_LOADED = False


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


def generate_text_local(prompt, max_tokens=200, temperature=0.7):
    """Generate text using local Sarvam model"""
    global tokenizer, model, MODEL_LOADED
    
    if not MODEL_LOADED:
        return {
            "success": False,
            "error": "Model not loaded",
            "text": prompt  # Return original text
        }
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return {
            "success": True,
            "text": generated_text,
            "model": SARVAM_MODEL
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": prompt
        }


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
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model": SARVAM_MODEL,
        "model_loaded": MODEL_LOADED,
        "device": device
    }


@app.post('/generate')
def generate(request: GenerateRequest):
    """Generate text using Sarvam AI"""
    try:
        result = generate_text_local(
            request.prompt,
            request.max_tokens,
            request.temperature
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/correct-tamil')
def correct_tamil(request: CorrectTamilRequest):
    """Correct Tamil OCR text"""
    try:
        prompt = f"Correct the following Tamil text for OCR errors:\n\nText: {request.text}\n\nCorrected:"
        
        result = generate_text_local(prompt, max_tokens=500, temperature=0.3)
        
        if result.get("success"):
            return {
                "success": True,
                "original": request.text,
                "corrected": result.get("text", request.text).strip(),
                "model": SARVAM_MODEL
            }
        
        # Return original text if correction fails
        return {
            "success": False,
            "error": result.get("error", "Correction failed"),
            "original": request.text,
            "corrected": request.text
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original": request.text,
            "corrected": request.text
        }


@app.post('/translate')
def translate(request: TranslateRequest):
    """Translate text"""
    try:
        langs = {'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi'}
        
        prompt = f"Translate from {langs.get(request.source_lang, request.source_lang)} to {langs.get(request.target_lang, request.target_lang)}: {request.text}\n\nTranslation:"
        
        result = generate_text_local(prompt, max_tokens=300, temperature=0.3)
        
        if result.get("success"):
            return {
                "success": True,
                "original": request.text,
                "translated": result.get("text", "")
            }
        
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\nStarting Sarvam AI Backend Server (FastAPI)")
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
