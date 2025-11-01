# ☁️ Cloud API Integration Guide

## Optional: Using Google Cloud Vision API Instead of Local Model

This guide explains how to integrate cloud-based APIs for coconut purity classification instead of using a local `.h5` model file.

---

## 🎯 Why Use Cloud APIs?

### Advantages:
- **No Model Training Required**: Use pre-trained cloud models
- **Automatic Updates**: Cloud providers continuously improve models
- **Scalability**: Handle high traffic without local GPU
- **Lower Storage**: No need to store large model files

### Disadvantages:
- **Cost**: Pay per API call
- **Internet Required**: Must have active connection
- **Latency**: Network overhead for each request
- **Privacy**: Images sent to third-party servers

---

## 🔧 Google Cloud Vision API Integration

### Step 1: Setup Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Cloud Vision API**
4. Create service account credentials
5. Download JSON key file

### Step 2: Install Google Cloud Library

```bash
pip install google-cloud-vision
```

Update `requirements.txt`:
```
flask==2.3.3
tensorflow==2.13.0
opencv-python==4.8.0.76
numpy==1.24.3
pillow==10.0.0
werkzeug==2.3.7
google-cloud-vision==3.4.0
```

### Step 3: Set Environment Variable

```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\service-account-key.json

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

### Step 4: Create Cloud Vision Integration

Create a new file `cloud_predict.py`:

```python
"""
Cloud-based prediction using Google Cloud Vision API
"""

from google.cloud import vision
import io

def classify_coconut_google(image_path):
    """
    Classify coconut using Google Cloud Vision API
    
    Args:
        image_path (str): Path to coconut image
        
    Returns:
        tuple: (predicted_label, confidence_percentage)
    """
    # Initialize Vision API client
    client = vision.ImageAnnotatorClient()
    
    # Load image
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # Analyze labels to determine purity
    purity_score = analyze_labels(labels)
    
    return purity_score

def analyze_labels(labels):
    """
    Analyze Vision API labels to determine coconut purity
    
    Args:
        labels: List of label annotations from Vision API
        
    Returns:
        tuple: (purity_level, confidence)
    """
    # Extract label descriptions and scores
    label_data = [(label.description.lower(), label.score) for label in labels]
    
    # Keywords for purity classification
    high_purity_keywords = ['clean', 'fresh', 'white', 'pure', 'smooth', 'intact']
    medium_purity_keywords = ['brown', 'natural', 'organic', 'dry']
    low_purity_keywords = ['cracked', 'damaged', 'dark', 'rough', 'old', 'weathered']
    
    # Calculate scores
    high_score = 0
    medium_score = 0
    low_score = 0
    
    for desc, score in label_data:
        if any(keyword in desc for keyword in high_purity_keywords):
            high_score += score
        elif any(keyword in desc for keyword in low_purity_keywords):
            low_score += score
        elif any(keyword in desc for keyword in medium_purity_keywords):
            medium_score += score
    
    # Determine purity level
    scores = {
        'High Purity': high_score,
        'Medium Purity': medium_score,
        'Low Purity': low_score
    }
    
    # Get highest score
    purity_level = max(scores, key=scores.get)
    confidence = min(scores[purity_level] * 100, 99.9)  # Cap at 99.9%
    
    # Default to medium if all scores are low
    if confidence < 30:
        purity_level = 'Medium Purity'
        confidence = 65.0
    
    return purity_level, round(confidence, 2)
```

### Step 5: Update `predict.py`

Add cloud option to `predict.py`:

```python
"""
Coconut Purity Prediction Module
Supports both local model and cloud API
"""

import numpy as np
import os

# Configuration
USE_CLOUD_API = os.getenv('USE_CLOUD_API', 'False').lower() == 'true'

# Check if TensorFlow is available
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available.")

# Check if Cloud Vision is available
try:
    from cloud_predict import classify_coconut_google
    CLOUD_VISION_AVAILABLE = True
except ImportError:
    CLOUD_VISION_AVAILABLE = False
    if USE_CLOUD_API:
        print("⚠️ Google Cloud Vision not available.")

classes = ['High Purity', 'Medium Purity', 'Low Purity']
MODEL_PATH = 'model/coconut_purity_model.h5'

model = None

def load_purity_model():
    """Load the pre-trained model if available"""
    global model
    if USE_CLOUD_API:
        print("ℹ️ Using Google Cloud Vision API for predictions")
        return True
    
    if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    else:
        print("ℹ️ Using simulation mode for demonstration")
        return False

def predict_purity(img_path):
    """
    Predict purity using cloud API or local model
    """
    # Use Cloud API if enabled
    if USE_CLOUD_API and CLOUD_VISION_AVAILABLE:
        try:
            return classify_coconut_google(img_path)
        except Exception as e:
            print(f"Cloud API error: {e}")
            print("Falling back to local prediction")
    
    # Use local model
    if model is not None and TENSORFLOW_AVAILABLE:
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            
            preds = model.predict(x, verbose=0)
            predicted_class = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            return classes[predicted_class], round(confidence, 2)
        except Exception as e:
            print(f"Prediction error: {e}")
            return simulate_prediction(img_path)
    else:
        return simulate_prediction(img_path)

def simulate_prediction(img_path):
    """Simulate prediction for demo"""
    filename = os.path.basename(img_path).lower()
    
    if 'high' in filename or 'good' in filename or 'pure' in filename:
        label = classes[0]
        confidence = np.random.uniform(85, 98)
    elif 'low' in filename or 'bad' in filename or 'poor' in filename:
        label = classes[2]
        confidence = np.random.uniform(75, 92)
    elif 'medium' in filename or 'mid' in filename:
        label = classes[1]
        confidence = np.random.uniform(80, 95)
    else:
        np.random.seed(hash(img_path) % (2**32))
        label = np.random.choice(classes)
        confidence = np.random.uniform(70, 95)
    
    return label, round(confidence, 2)

load_purity_model()
```

### Step 6: Enable Cloud API

Set environment variable before running:

```bash
# Windows
set USE_CLOUD_API=True
python app.py

# Linux/Mac
export USE_CLOUD_API=True
python app.py
```

---

## 💰 Cost Estimation

### Google Cloud Vision API Pricing (as of 2024):

| Feature | First 1,000 units/month | 1,001 - 5,000,000 units |
|---------|------------------------|------------------------|
| Label Detection | Free | $1.50 per 1,000 images |

**Example Costs:**
- 100 images/day = 3,000/month = **$3.00/month**
- 500 images/day = 15,000/month = **$21.00/month**
- 1,000 images/day = 30,000/month = **$43.50/month**

---

## 🔄 Alternative Cloud APIs

### AWS Rekognition

```python
import boto3

def classify_coconut_aws(image_path):
    """Use AWS Rekognition for classification"""
    client = boto3.client('rekognition')
    
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    
    response = client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=70
    )
    
    # Analyze labels similar to Google Vision
    labels = [(label['Name'].lower(), label['Confidence']/100) 
              for label in response['Labels']]
    
    return analyze_labels_aws(labels)
```

### Azure Computer Vision

```python
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

def classify_coconut_azure(image_path):
    """Use Azure Computer Vision for classification"""
    subscription_key = "YOUR_SUBSCRIPTION_KEY"
    endpoint = "YOUR_ENDPOINT"
    
    client = ComputerVisionClient(
        endpoint, 
        CognitiveServicesCredentials(subscription_key)
    )
    
    with open(image_path, 'rb') as image_stream:
        tags_result = client.tag_image_in_stream(image_stream)
    
    labels = [(tag.name.lower(), tag.confidence) 
              for tag in tags_result.tags]
    
    return analyze_labels_azure(labels)
```

---

## 🔒 Security Best Practices

### 1. Protect API Keys

**Never hardcode credentials!**

```python
# ❌ BAD
api_key = "AIzaSyD1234567890abcdefg"

# ✅ GOOD
import os
api_key = os.getenv('GOOGLE_API_KEY')
```

### 2. Use Environment Variables

Create `.env` file:
```
GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json
USE_CLOUD_API=True
```

Install python-dotenv:
```bash
pip install python-dotenv
```

Load in `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Add to `.gitignore`

```
.env
credentials/
*.json
```

---

## 📊 Comparison: Local vs Cloud

| Feature | Local Model | Cloud API |
|---------|-------------|-----------|
| **Setup Complexity** | High (model training) | Low (API setup) |
| **Cost** | Free (after training) | Pay per use |
| **Speed** | Fast (local) | Slower (network) |
| **Accuracy** | Custom trained | General purpose |
| **Privacy** | High (data stays local) | Lower (sent to cloud) |
| **Scalability** | Limited by hardware | Unlimited |
| **Maintenance** | Manual updates | Automatic |
| **Internet Required** | No | Yes |

---

## 🎯 Recommendation

### Use Local Model When:
- You have specific coconut dataset
- Privacy is critical
- High volume (>10,000 images/month)
- Offline operation needed

### Use Cloud API When:
- Quick prototype needed
- Low volume (<1,000 images/month)
- No ML expertise available
- Want automatic improvements

### Hybrid Approach:
- Use cloud API for initial launch
- Collect data and feedback
- Train custom model later
- Switch to local model for production

---

## 🧪 Testing Cloud Integration

### Test Script

```python
# test_cloud.py
from cloud_predict import classify_coconut_google

# Test with sample image
result = classify_coconut_google('test_coconut.jpg')
print(f"Purity: {result[0]}")
print(f"Confidence: {result[1]}%")
```

Run test:
```bash
python test_cloud.py
```

---

## 📚 Additional Resources

- [Google Cloud Vision Documentation](https://cloud.google.com/vision/docs)
- [AWS Rekognition Documentation](https://docs.aws.amazon.com/rekognition/)
- [Azure Computer Vision Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/)

---

**Last Updated**: 2024
