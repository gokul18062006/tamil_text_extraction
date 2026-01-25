"""
Tamil Handwritten Text Recognition System
Converts Tamil handwritten images to text using:
- PaddleOCR (Deep learning OCR for Tamil text extraction)
- Sarvam AI (Tamil text correction)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
import io
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Tamil Handwritten Text Recognition",
    description="Convert Tamil handwritten images to text using PaddleOCR and Sarvam AI"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PaddleOCR Configuration
print("Loading PaddleOCR reader for Tamil...")
try:
    # Initialize PaddleOCR with Tamil language (CPU mode)
    ocr = PaddleOCR(lang='ta', use_textline_orientation=True)
    print("[OK] PaddleOCR loaded successfully with Tamil support!")
except Exception as e:
    print(f"[ERROR] Error loading PaddleOCR: {e}")
    ocr = None

# Sarvam API Configuration
SARVAM_BACKEND_URL = os.getenv("SARVAM_BACKEND_URL", "http://localhost:5000")

# Request Models
class TextRequest(BaseModel):
    text: str
    use_sarvam: bool = True


def extract_text_from_image(image: Image.Image, lang: str = None) -> str:
    """Extract Tamil text from image using PaddleOCR"""
    try:
        print(f"[DEBUG] Starting PaddleOCR extraction. Image mode: {image.mode}, Size: {image.size}")
        
        if ocr is None:
            print("[ERROR] OCR is None!")
            raise HTTPException(status_code=500, detail="PaddleOCR not loaded")
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        print(f"[DEBUG] Converted to numpy array. Shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        # Perform OCR with PaddleOCR
        print("[DEBUG] Calling ocr.ocr()...")
        result = ocr.ocr(image_np, cls=True)
        print(f"[DEBUG] PaddleOCR results: {result}")
        
        # Extract text from results
        # PaddleOCR returns: [[line1, line2, ...]] where each line is [bbox, (text, confidence)]
        extracted_texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0]  # text is at index 1, element 0
                    extracted_texts.append(text)
        
        extracted_text = ' '.join(extracted_texts)
        print(f"[DEBUG] Extracted text: {extracted_text}")
        
        return extracted_text.strip() if extracted_text else ""
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in extract_text_from_image: {type(e).__name__}: {str(e)}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"PaddleOCR Error: {str(e)}")


def correct_with_sarvam(text: str) -> dict:
    """Correct Tamil text using Sarvam AI"""
    try:
        response = requests.post(
            f"{SARVAM_BACKEND_URL}/correct-tamil",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"Sarvam API error: {response.status_code}",
                "original": text,
                "corrected": text
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Sarvam connection error: {str(e)}",
            "original": text,
            "corrected": text
        }


@app.get("/")
def home():
    return {
        "service": "Tamil Handwritten Text Recognition",
        "status": "running",
        "components": {
            "easyocr": "Deep learning OCR for Tamil text extraction",
            "sarvam": "Tamil text correction"
        },
        "endpoints": {
            "POST /process-image": "Process Tamil handwritten image",
            "POST /process-with-easyocr": "Only EasyOCR extraction",
            "POST /process-with-sarvam": "EasyOCR + Sarvam correction",
            "POST /correct-text": "Correct extracted text",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Check system health"""
    status = {
        "easyocr": reader is not None,
        "sarvam": False
    }
    
    # Check Sarvam
    try:
        response = requests.get(f"{SARVAM_BACKEND_URL}/health", timeout=5)
        status["sarvam"] = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if all(status.values()) else "degraded",
        "components": status
    }


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    use_sarvam: bool = True
):
    """Process Tamil handwritten image - Main endpoint"""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Step 1: Extract text with EasyOCR
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text:
            return JSONResponse({
                "success": False,
                "error": "No text detected in image",
                "easyocr": {"text": ""}
            })
        
        result = {
            "success": True,
            "easyocr": {
                "text": extracted_text,
                "status": "completed"
            }
        }
        
        current_text = extracted_text
        
        # Step 2: Correct with Sarvam (if enabled)
        if use_sarvam:
            sarvam_result = correct_with_sarvam(current_text)
            result["sarvam"] = sarvam_result
            if sarvam_result.get("success"):
                current_text = sarvam_result.get("corrected", current_text)
        
        result["final_text"] = current_text
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-with-paddleocr")
async def process_with_paddleocr(file: UploadFile = File(...)):
    """Process image using only PaddleOCR"""
    try:
        if ocr is None:
            raise HTTPException(status_code=500, detail="PaddleOCR not initialized. Please check server logs.")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        extracted_text = extract_text_from_image(image)
        
        return {
            "success": True,
            "method": "paddleocr_only",
            "text": extracted_text
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/process-with-sarvam")
async def process_with_sarvam(file: UploadFile = File(...)):
    """Process image using EasyOCR + Sarvam"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text
        extracted_text = extract_text_from_image(image)
        
        # Correct with Sarvam
        sarvam_result = correct_with_sarvam(extracted_text)
        
        return {
            "success": True,
            "method": "easyocr_sarvam",
            "easyocr_text": extracted_text,
            "sarvam_result": sarvam_result,
            "final_text": sarvam_result.get("corrected", extracted_text)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    port = int(os.getenv("MAIN_PORT", 8000))
    
    print("\n" + "=" * 80)
    print("Tamil Handwritten Text Recognition System")
    print("=" * 80)
    print(f"Server: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 80)
    print("\nComponents:")
    print("  [+] EasyOCR - Deep learning Tamil text extraction")
    print("  [+] Sarvam AI - Tamil text correction")
    print("\nMain Endpoints:")
    print("  POST /process-image         - EasyOCR extraction")
    print("  POST /process-with-easyocr  - EasyOCR extraction only")
    print("  POST /process-with-sarvam   - EasyOCR + Sarvam correction (recommended)")
    print("  POST /correct-text          - Correct already extracted text")
    print("\nTest at: http://localhost:" + str(port) + "/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")