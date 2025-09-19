"""
FastAPI Prediction API for Autism Detection
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from torchvision import transforms

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autism_model import LightweightAutismCNN

# Initialize FastAPI app
app = FastAPI(
    title="Autism Detection API",
    description="API for detecting autism from facial images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
transform = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_autistic: float
    probability_non_autistic: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

def load_model(model_path: str):
    """Load the trained model"""
    global model, device, transform
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model = LightweightAutismCNN(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Set up transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

def predict_image(image_tensor: torch.Tensor) -> dict:
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction
            _, predicted = torch.max(outputs, 1)
            prediction = "autistic" if predicted.item() == 0 else "non_autistic"
            
            # Get confidence scores
            confidence = torch.max(probabilities).item()
            prob_autistic = probabilities[0][0].item()
            prob_non_autistic = probabilities[0][1].item()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probability_autistic": prob_autistic,
                "probability_non_autistic": prob_non_autistic
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = Path(__file__).parent.parent.parent / "model_artifacts" / "autism_detection_model.pth"
    
    if not model_path.exists():
        print(f"⚠️ Model file not found at {model_path}")
        print("Please train the model first by running: python src/train_model.py")
        return
    
    success = load_model(str(model_path))
    if not success:
        print("❌ Failed to load model")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Autism Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_base64": "/predict_base64",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Predict autism from uploaded image file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type - check both content_type and file extension
    is_image = False
    if file.content_type and file.content_type.startswith('image/'):
        is_image = True
    elif file.filename:
        # Check file extension as fallback
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext in valid_extensions:
            is_image = True
    
    if not is_image:
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        result = predict_image(image_tensor)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_from_base64(data: dict):
    """Predict autism from base64 encoded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request body")
        
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        result = predict_image(image_tensor)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_architecture": "LightweightAutismCNN",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,
        "input_size": "224x224x3",
        "num_classes": 2,
        "classes": ["autistic", "non_autistic"],
        "device": str(device)
    }

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
