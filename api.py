"""
FastAPI Web Service for Beauty Prediction
Production-ready REST API for serving beauty prediction model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
from pathlib import Path

from model import BeautyPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Beauty Prediction API",
    description="Real-time facial beauty prediction using MobileNetV3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global model instance
model = None
device = None
face_cascade = None

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    beauty_score: float
    confidence: float
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str


def load_model():
    """Load the beauty prediction model"""
    global model, device, face_cascade
    
    logger.info("Loading model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = BeautyPredictor(pretrained=True)
    model.to(device)
    model.eval()
    logger.info("✓ Model loaded successfully")
    
    # Load face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.error("Failed to load face cascade!")
        raise RuntimeError("Face detector initialization failed")
    logger.info("✓ Face detector loaded")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        load_model()
        logger.info("API ready to serve requests")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    try:
        # Read image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


def detect_face(image: np.ndarray) -> Optional[tuple]:
    """Detect face in image and return crop"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        
        # Apply 20% padding
        frame_h, frame_w = image.shape[:2]
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame_w, x + w + pad_w)
        y2 = min(frame_h, y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        
        # Calculate confidence (rough estimate based on face size)
        face_area = w * h
        frame_area = frame_w * frame_h
        confidence = min(0.99, 0.5 + (face_area / frame_area) * 2)
        
        return face_crop, confidence
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return None


def preprocess_face(face_img: np.ndarray) -> torch.Tensor:
    """Preprocess face for model inference"""
    # Resize to 224x224
    img = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img = (img - MEAN) / STD
    
    # Convert to CHW and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float()
    
    return img_tensor


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web UI"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
    <html>
        <head><title>Beauty Prediction API</title></head>
        <body>
            <h1>Beauty Prediction API</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    param_count = sum(p.numel() for p in model.parameters())
    
    return {
        "model_name": "MobileNetV3-Large Beauty Predictor",
        "version": "1.0.0",
        "parameters": param_count,
        "input_size": [224, 224],
        "output_range": [1.0, 5.0],
        "device": str(device),
        "face_detector": "OpenCV Haar Cascade"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict beauty score from uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        PredictionResponse with beauty score and metadata
    """
    start_time = time.time()
    
    # Validate model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Detect face
        result = detect_face(image)
        if result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face_crop, confidence = result
        
        # Preprocess for model
        face_tensor = preprocess_face(face_crop).to(device)
        
        # Predict
        with torch.no_grad():
            score = model.predict(face_tensor)
        
        beauty_score = float(score.cpu().item())
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Prediction: {beauty_score:.2f}, Time: {processing_time:.2f}ms")
        
        return {
            "beauty_score": beauty_score,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "model_version": "1.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict beauty scores for multiple images
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            response = await predict(file)
            results.append({
                "filename": file.filename,
                "success": True,
                **response.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
