"""
Test the trained copra grading model with sample images
"""

import os
import sys
import random
from pathlib import Path

print("=" * 70)
print("🥥 COPRA GRADING MODEL - PREDICTION TEST")
print("=" * 70)

# Import prediction module
from predict import load_purity_model, predict_purity

# Load the model
print("\n📦 Loading trained model...")
if not load_purity_model():
    print("❌ Failed to load model!")
    sys.exit(1)

print("\n✅ Model loaded successfully!")

# Get test images from the test dataset
test_dir = Path(r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized\test")

if not test_dir.exists():
    print(f"❌ Test directory not found: {test_dir}")
    sys.exit(1)

# Get all classes
classes = [d for d in test_dir.iterdir() if d.is_dir()]
print(f"\n📂 Found {len(classes)} classes: {[c.name for c in classes]}")

# Test predictions
print("\n" + "=" * 70)
print("🔍 TESTING PREDICTIONS")
print("=" * 70)

total_correct = 0
total_tested = 0

for class_dir in sorted(classes):
    class_name = class_dir.name
    images = list(class_dir.glob("*.jpg"))[:5]  # Test 5 images per class
    
    print(f"\n📊 Testing {class_name} ({len(images)} images):")
    print("-" * 70)
    
    correct = 0
    for img_path in images:
        predicted_label, confidence = predict_purity(str(img_path))
        is_correct = predicted_label == class_name
        
        if is_correct:
            correct += 1
            total_correct += 1
        
        total_tested += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_path.name[:40]:40} | Predicted: {predicted_label:10} | Confidence: {confidence:5.1f}% | Actual: {class_name}")
    
    accuracy = (correct / len(images)) * 100 if images else 0
    print(f"\n   Class Accuracy: {correct}/{len(images)} ({accuracy:.1f}%)")

# Overall accuracy
print("\n" + "=" * 70)
print("📈 OVERALL RESULTS")
print("=" * 70)
overall_accuracy = (total_correct / total_tested) * 100 if total_tested > 0 else 0
print(f"Total Tested: {total_tested}")
print(f"Correct Predictions: {total_correct}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print("=" * 70)

# Test with random images
print("\n" + "=" * 70)
print("🎲 RANDOM IMAGE PREDICTIONS")
print("=" * 70)

all_images = []
for class_dir in classes:
    all_images.extend([(img, class_dir.name) for img in class_dir.glob("*.jpg")])

random.shuffle(all_images)
sample_images = all_images[:10]  # Test 10 random images

for img_path, actual_class in sample_images:
    predicted_label, confidence = predict_purity(str(img_path))
    is_correct = predicted_label == actual_class
    status = "✅" if is_correct else "❌"
    
    print(f"{status} Predicted: {predicted_label:10} ({confidence:5.1f}%) | Actual: {actual_class:10} | {img_path.name[:40]}")

print("\n" + "=" * 70)
print("✅ TESTING COMPLETE!")
print("=" * 70)
