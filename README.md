# Tamil Handwritten Text Recognition System

Convert Tamil handwritten images to text using Tesseract OCR and Sarvam AI.

## 🎯 Features

- **Tesseract OCR** - Extracts Tamil text from handwritten images
- **Sarvam AI** - Corrects OCR errors and improves accuracy
- **FastAPI** - Modern REST API with automatic documentation
- **Multi-language Support** - Tamil + English

## 📋 Components

1. **Main Application (Port 8000)** - Tamil OCR processing
2. **Sarvam Backend (Port 5000)** - AI text correction

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Tesseract OCR ([Download](https://github.com/UB-Mannheim/tesseract/wiki))
- Virtual environment

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd new_project
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**
```powershell
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**
- Copy `backend/.env` and update if needed
- Set your HuggingFace token in `.env`

## 🏃 Running the Application

### Start Main Application (Port 8000)
```bash
python main.py
```

### Start Sarvam Backend (Port 5000)
```bash
cd backend
python backend_sarvam.py
```

## 📖 API Documentation

Once running, visit:
- Main API: http://localhost:8000/docs
- Sarvam Backend: http://localhost:5000/docs

## 🌐 Endpoints

### Main Application

- `POST /process-image` - Process Tamil handwritten image
- `POST /process-with-sarvam` - Tesseract + Sarvam (best accuracy)
- `POST /process-with-tesseract` - Tesseract only
- `POST /correct-text` - Correct already extracted text
- `GET /health` - Health check

## 🛠️ Technology Stack

- **FastAPI** - Web framework
- **Tesseract OCR** - Text extraction
- **Sarvam AI** - Tamil text correction
- **Pillow** - Image processing
- **Uvicorn** - ASGI server

## 📝 License

MIT License

## 👨‍💻 Author

Gokul P
