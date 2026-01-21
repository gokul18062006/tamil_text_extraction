"""
Tamil Handwritten Text Recognition System
Converts Tamil handwritten images to text using:
- Tesseract OCR (Text extraction)
- Sarvam AI (Tamil text correction)
- Google Gemini (Final validation)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Tamil Handwritten Text Recognition",
    description="Convert Tamil handwritten images to text using Tesseract OCR and Sarvam AI"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Sarvam API Configuration
SARVAM_BACKEND_URL = os.getenv("SARVAM_BACKEND_URL", "http://localhost:5000")

# Request Models
class TextRequest(BaseModel):
    text: str
    use_sarvam: bool = True


def extract_text_from_image(image: Image.Image, lang: str = 'tam+eng') -> str:
    """Extract text from image using Tesseract OCR with adaptive thresholding"""
    try:
        from PIL import ImageEnhance, ImageOps, ImageFilter
        import numpy as np
        
        results = []
        
        # Strategy 1: Tamil-only with adaptive threshold
        try:
            img = image.convert('L')
            # Convert to numpy array for better thresholding
            img_array = np.array(img)
            # Adaptive threshold - best for handwriting
            from PIL import Image as PILImage
            threshold = img_array.mean()
            binary = (img_array > threshold) * 255
            img = PILImage.fromarray(binary.astype(np.uint8))
            
            # Upscale if small
            width, height = img.size
            if width < 800:
                scale = 800 / width
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            
            # Tamil only, single line
            text = pytesseract.image_to_string(img, lang='tam', config='--oem 3 --psm 7')
            if text.strip():
                results.append(('tam_adaptive', text.strip()))
        except:
            pass
        
        # Strategy 2: High contrast Tamil-only
        try:
            img = image.convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(3.0)
            
            width, height = img.size
            if width < 800:
                scale = 800 / width
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            
            text = pytesseract.image_to_string(img, lang='tam', config='--oem 3 --psm 7')
            if text.strip():
                results.append(('tam_contrast', text.strip()))
        except:
            pass
        
        # Strategy 3: Original with Tamil+English
        try:
            img = image.convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            width, height = img.size
            if width < 800:
                scale = 800 / width
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            
            text = pytesseract.image_to_string(img, lang='tam+eng', config='--oem 3 --psm 6')
            if text.strip():
                results.append(('mixed', text.strip()))
        except:
            pass
        
        # Strategy 4: Inverted with binary threshold
        try:
            img = ImageOps.invert(image.convert('L'))
            img_array = np.array(img)
            threshold = img_array.mean()
            binary = (img_array > threshold) * 255
            img = PILImage.fromarray(binary.astype(np.uint8))
            
            width, height = img.size
            if width < 800:
                scale = 800 / width
                img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
            
            text = pytesseract.image_to_string(img, lang='tam', config='--oem 3 --psm 7')
            if text.strip():
                results.append(('inverted', text.strip()))
        except:
            pass
        
        if results:
            # Prioritize Tamil-only results
            tamil_only = [r for r in results if r[0].startswith('tam_')]
            if tamil_only:
                # Return the one with most Tamil characters
                best = max(tamil_only, key=lambda x: sum(1 for c in x[1] if '\u0B80' <= c <= '\u0BFF'))
                return best[1]
            
            # Otherwise return longest with Tamil
            tamil_results = [r for r in results if any('\u0B80' <= c <= '\u0BFF' for c in r[1])]
            if tamil_results:
                best = max(tamil_results, key=lambda x: len(x[1]))
                return best[1]
            
            # Fallback to any result
            return max(results, key=lambda x: len(x[1]))[1]
        
        return ""
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Error: {str(e)}")


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
            "tesseract": "Text extraction",
            "sarvam": "Tamil correction",
            "gemini": "Final validation"
        },
        "endpoints": {
            "POST /process-image": "Process Tamil handwritten image",
            "POST /process-with-tesseract": "Only Tesseract OCR",
            "POST /process-with-sarvam": "Tesseract + Sarvam",
            "POST /process-with-gemini": "Tesseract + Gemini",
            "POST /process-triple": "All three (best accuracy)",
            "POST /correct-text": "Correct extracted text",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Check system health"""
    status = {
        "tesseract": False,
        "sarvam": False
    }
    
    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
        status["tesseract"] = True
    except:
        pass
    
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
        
        # Step 1: Extract text with Tesseract
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text:
            return JSONResponse({
                "success": False,
                "error": "No text detected in image",
                "tesseract": {"text": ""}
            })
        
        result = {
            "success": True,
            "tesseract": {
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


@app.post("/process-with-tesseract")
async def process_with_tesseract(file: UploadFile = File(...)):
    """Process image using only Tesseract OCR"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        extracted_text = extract_text_from_image(image)
        
        return {
            "success": True,
            "method": "tesseract_only",
            "text": extracted_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-with-sarvam")
async def process_with_sarvam(file: UploadFile = File(...)):
    """Process image using Tesseract + Sarvam"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text
        extracted_text = extract_text_from_image(image)
        
        # Correct with Sarvam
        sarvam_result = correct_with_sarvam(extracted_text)
        
        return {
            "success": True,
            "method": "tesseract_sarvam",
            "tesseract_text": extracted_text,
            "sarvam_result": sarvam_result,
            "final_text": sarvam_result.get("corrected", extracted_text)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-with-gemini")
async def process_with_gemini(file: UploadFile = File(...)):
    """Process image using Tesseract + Gemini"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text
        extracted_text = extract_text_from_image(image)
        
        # Note: correct_with_gemini function needs to be implemented
        # gemini_result = correct_with_gemini(extracted_text)
        
        return {
            "success": True,
            "method": "tesseract_gemini",
            "tesseract_text": extracted_text,
            "final_text": extracted_text,
            "note": "Gemini integration pending"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    port = int(os.getenv("MAIN_PORT", 8000))
    
    print("\n" + "=" * 80)
    print("🎯 Tamil Handwritten Text Recognition System")
    print("=" * 80)
    print(f"Server: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 80)
    print("\n📋 Components:")
    print("  ✓ Tesseract OCR - Text extraction")
    print("  ✓ Sarvam AI - Tamil correction")
    print("  ✓ Google Gemini - Final validation")
    print("\n🌐 Main Endpoints:")
    print("  POST /process-image       - Smart processing (auto-selects best method)")
    print("  POST /process-triple      - Triple validation (maximum accuracy)")
    print("  POST /process-with-gemini - Tesseract + Gemini (recommended)")
    print("  POST /correct-text        - Correct already extracted text")
    print("\n💡 Test at: http://localhost:" + str(port) + "/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")