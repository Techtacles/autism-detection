"""
Test Client for Autism Detection API
"""

import requests
import base64
from PIL import Image
import io
import json
from pathlib import Path

class AutismDetectionClient:
    """Client for testing the Autism Detection API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_file(self, image_path):
        """Predict from image file"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/predict", files=files)
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_base64(self, image_path):
        """Predict from base64 encoded image"""
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Make request
            data = {"image": base64_image}
            response = requests.post(f"{self.base_url}/predict_base64", json=data)
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model_info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def test_api():
    """Test the API with sample images"""
    client = AutismDetectionClient()
    
    print("🧪 Testing Autism Detection API")
    print("=" * 40)
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if "error" in health:
        print("❌ API is not running. Please start the API first.")
        return
    
    # Model info
    print("\n2. Model Information:")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # Test with sample images
    print("\n3. Testing with sample images:")
    
    # Find test images
    data_dir = Path("/Users/OFFISONG_1/Desktop/COMPANIES/AIRLAB_IT/autism-detection/data/processed/test")
    
    if data_dir.exists():
        # Test with autistic image
        autistic_dir = data_dir / "autistic"
        if autistic_dir.exists():
            autistic_images = list(autistic_dir.glob("*.jpg"))[:2]
        for img_path in autistic_images:
            print(f"\n   Testing with {img_path.name} (should be autistic):")
            result = client.predict_from_file(str(img_path))
            if 'error' in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Prediction: {result.get('prediction', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Probability (autistic): {result.get('probability_autistic', 0):.3f}")
        
        # Test with non-autistic image
        non_autistic_dir = data_dir / "non_autistic"
        if non_autistic_dir.exists():
            non_autistic_images = list(non_autistic_dir.glob("*.jpg"))[:2]
        for img_path in non_autistic_images:
            print(f"\n   Testing with {img_path.name} (should be non-autistic):")
            result = client.predict_from_file(str(img_path))
            if 'error' in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Prediction: {result.get('prediction', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Probability (non-autistic): {result.get('probability_non_autistic', 0):.3f}")
    else:
        print("   No test images found. Please run the training script first.")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()
