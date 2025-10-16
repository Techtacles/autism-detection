# 🧠 Autism Detection from Facial Images V2

A lightweight deep learning model for detecting autism spectrum disorder from facial images using PyTorch and FastAPI. This project provides a complete end-to-end solution for autism detection with a production-ready API.

## 🎯 Project Overview

This project implements a lightweight CNN model to classify facial images as either "autistic" or "non-autistic" with high accuracy. The model is designed to be efficient, deployable, and suitable for real-world applications while maintaining ethical considerations for medical AI.

## ✨ Key Features

- **Lightweight Model**: Only 422K parameters (1.6MB)
- **High Accuracy**: 73.45% test accuracy with balanced precision/recall
- **FastAPI Integration**: Production-ready REST API
- **Multiple Input Formats**: File upload and base64 encoding
- **Comprehensive Analysis**: Data preprocessing and visualization
- **Easy Deployment**: Docker-ready with uv package management

## 📁 Project Structure

```
autism-detection/
├── src/
│   ├── data/
│   │   └── data_analysis.py          # Data analysis and preprocessing
│   ├── models/
│   │   └── autism_model.py           # Model architecture and training
│   ├── api/
│   │   ├── prediction_api.py         # FastAPI prediction service
│   │   └── test_client.py            # API testing client
│   └── train_model.py                # Main training script
├── data/
│   └── processed/                    # Consolidated dataset
│       ├── train/
│       │   ├── autistic/             # 864 images
│       │   └── non_autistic/         # 873 images
│       ├── test/
│       │   ├── autistic/             # 289 images
│       │   └── non_autistic/         # 291 images
│       └── val/
│           ├── autistic/             # 289 images
│           └── non_autistic/         # 291 images
├── model_artifacts/                  # Trained model and reports
│   ├── autism_detection_model.pth    # Trained model weights
│   ├── classification_report.txt     # Detailed performance metrics
│   ├── confusion_matrix.png          # Confusion matrix visualization
│   ├── training_history.png          # Training progress plots
│   └── model_info.txt               # Model specifications
├── pyproject.toml                    # uv project configuration
├── requirements.txt                  # pip requirements (backup)
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Or pip (alternative)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autism-detection

# Install dependencies with uv (recommended)
uv sync

# Or install with pip
pip install -r requirements.txt
```

### 1. Data Preparation

The project automatically consolidates multiple datasets into a unified structure:

```bash
# Analyze and preprocess datasets
uv run python src/data/data_analysis.py
```

### 2. Model Training

Train the lightweight CNN model:

```bash
# Train the model (this will also run data analysis)
uv run python src/train_model.py
```

**Training Results:**
- **Model Size**: 1.6MB (422K parameters)
- **Test Accuracy**: 73.45%
- **Training Time**: ~5-10 minutes on CPU
- **Best Validation Accuracy**: 76.21%

### 3. API Deployment

Start the FastAPI prediction service:

```bash
# Start the API server
uv run python src/api/prediction_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Test the API

```bash
# Test the API with sample images
uv run python src/api/test_client.py
```

## 🧠 Model Architecture

The model uses a lightweight CNN architecture optimized for efficiency:

### Architecture Details
- **Input**: 224×224×3 RGB images
- **Convolutional Layers**: 4 blocks with batch normalization
- **Pooling**: Max pooling + Global Average Pooling
- **Regularization**: Dropout (0.5)
- **Output**: 2 classes (autistic, non_autistic)
- **Parameters**: 422,530 trainable parameters
- **Model Size**: 1.6MB

### Performance Metrics
```
              precision    recall  f1-score   support

    autistic       0.68      0.87      0.77       289
non_autistic       0.82      0.60      0.69       291

    accuracy                           0.73       580
   macro avg       0.75      0.73      0.73       580
weighted avg       0.75      0.73      0.73       580
```

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```
Returns API status and model information.

### Predict from File Upload
```bash
POST /predict
Content-Type: multipart/form-data
Body: image file (jpg, png, etc.)
```

**Response:**
```json
{
  "prediction": "autistic",
  "confidence": 0.85,
  "probability_autistic": 0.85,
  "probability_non_autistic": 0.15
}
```

### Predict from Base64
```bash
POST /predict_base64
Content-Type: application/json
Body: {"image": "base64_encoded_image"}
```

### Model Information
```bash
GET /model_info
```
Returns detailed model specifications and performance metrics.

## 📈 Usage Examples

### Python Client

```python
import requests

# Upload image file
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Upload image
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"

# Get model info
curl http://localhost:8000/model_info
```

### JavaScript/Frontend

```javascript
// Upload image file
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
});
```

## 🛠️ Development

### Data Analysis

The `data_analysis.py` script provides comprehensive analysis:

- **Dataset Statistics**: Size, distribution, and quality metrics
- **Image Analysis**: Resolution, format, and preprocessing needs
- **Data Consolidation**: Merges multiple datasets into unified structure
- **Quality Assessment**: Identifies potential issues and improvements

### Model Training

The training process includes:

- **Data Augmentation**: Rotation, flipping, color jittering
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model during training
- **Comprehensive Evaluation**: Detailed metrics and visualizations

### API Features

- **FastAPI Framework**: High-performance async API
- **CORS Support**: Cross-origin requests for web applications
- **Multiple Input Formats**: File upload and base64 encoding
- **Error Handling**: Comprehensive error responses
- **Health Monitoring**: Real-time API status
- **Interactive Documentation**: Auto-generated API docs

## 📊 Dataset Information

### Consolidated Dataset
- **Total Images**: 2,897
- **Training Set**: 1,737 images (864 autistic, 873 non-autistic)
- **Test Set**: 580 images (289 autistic, 291 non-autistic)
- **Validation Set**: 580 images (289 autistic, 291 non-autistic)
- **Format**: JPG/PNG, 224×224 pixels
- **Balance**: ~50/50 split between classes

### Data Sources
The project consolidates multiple autism detection datasets:
- ASD Data (2,936 images)
- Autistic Children Dataset (2,926 images)
- Facial Dataset (2,897 images)

## 🔒 Ethical Considerations

⚠️ **Important Disclaimer**: This model is designed for research and educational purposes only.

### When using for clinical applications:
- Ensure proper validation with medical professionals
- Consider bias and fairness in model predictions
- Implement appropriate safeguards and disclaimers
- Follow ethical guidelines for medical AI applications
- Never use as a sole diagnostic tool

### Responsible AI Practices:
- Regular model auditing and bias testing
- Transparent reporting of limitations
- Clear documentation of training data sources
- Ongoing monitoring of model performance

## 📋 Requirements

### Core Dependencies
- Python 3.9+
- PyTorch 2.8.0+
- FastAPI 0.116.2+
- PIL/Pillow 11.3.0+
- scikit-learn 1.6.1+

### Optional Dependencies
- CUDA (for GPU acceleration)
- Docker (for containerization)

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
uv sync

# Start API
uv run python src/api/prediction_api.py
```

### Production Deployment

1. **Docker** (recommended):
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "src/api/prediction_api.py"]
```

2. **Cloud Platforms**:
   - AWS EC2/ECS
   - Google Cloud Run
   - Azure Container Instances
   - Heroku

3. **API Gateway Integration**:
   - AWS API Gateway
   - Google Cloud Endpoints
   - Azure API Management

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd autism-detection
uv sync

# Run tests
uv run python -m pytest tests/

# Format code
uv run black src/
uv run isort src/
```

## 📞 Support

- **Issues**: Open an issue in the repository
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `/docs` endpoint when API is running

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- Data usage agreements
- Ethical guidelines for medical AI
- Local regulations and privacy laws

## 🙏 Acknowledgments

- Dataset providers and researchers
- PyTorch and FastAPI communities
- Medical AI research community
- Open source contributors

---

**⚠️ Medical Disclaimer**: This tool is not intended for clinical diagnosis and should be used only for research and educational purposes. Always consult with qualified medical professionals for any health-related decisions.
