"""
Simple training script with visible progress for Copra grading
"""

import os
import sys
import numpy as np
from datetime import datetime

print("=" * 70)
print("🥥 COPRA GRADING MODEL TRAINING")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check TensorFlow
print("📦 Checking dependencies...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install tensorflow")
    sys.exit(1)

# Custom callback for progress
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch + 1}/{self.total_epochs}")
        print(f"{'='*70}")
        
    def on_epoch_end(self, epoch, logs=None):
        duration = (datetime.now() - self.epoch_start_time).total_seconds()
        percentage = ((epoch + 1) / self.total_epochs) * 100
        
        print(f"\n{'='*70}")
        print(f"✅ Epoch {epoch + 1}/{self.total_epochs} Complete ({percentage:.1f}%)")
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"📈 Training Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f"📈 Validation Accuracy: {logs.get('val_accuracy', 0):.4f}")
        print(f"📉 Training Loss: {logs.get('loss', 0):.4f}")
        print(f"📉 Validation Loss: {logs.get('val_loss', 0):.4f}")
        print(f"{'='*70}")

# Dataset path
dataset_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized\train"
print(f"\n📁 Dataset directory: {dataset_dir}")

if not os.path.exists(dataset_dir):
    print(f"❌ Dataset not found!")
    sys.exit(1)

# Check classes
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print(f"📂 Found {len(classes)} classes: {classes}")

for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   - {cls}: {count} images")

print("\n🔧 Preparing data generators...")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

img_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Batch size: {batch_size}")

# Create model
print("\n🏗️  Building model architecture...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")
print(f"📊 Total parameters: {model.count_params():,}")

# Create model directory
os.makedirs('model', exist_ok=True)

# Callbacks
epochs = 50
callbacks = [
    ProgressCallback(epochs),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('model/copra_model.h5', monitor='val_accuracy', 
                   save_best_only=True, save_weights_only=False, verbose=1)
]

print(f"\n🚀 Starting training for {epochs} epochs...")
print("=" * 70)

# Train
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=2  # One line per epoch
)

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"💾 Model saved to: model/copra_model.h5")

# Save class names
with open('model/class_names.txt', 'w') as f:
    for class_name in train_generator.class_indices.keys():
        f.write(f"{class_name}\n")
print(f"💾 Class names saved to: model/class_names.txt")

print(f"\n✅ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
