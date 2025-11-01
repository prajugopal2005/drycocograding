"""
Training script with real-time log file output
"""

import os
import sys
from datetime import datetime

# Setup logging to file
log_file = 'training_progress.log'

def log(message):
    """Write to both console and log file"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

# Clear previous log
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('')

log("=" * 70)
log("🥥 COPRA GRADING MODEL TRAINING")
log("=" * 70)

# Check TensorFlow
log("📦 Loading TensorFlow (this may take 30-60 seconds)...")
try:
    import tensorflow as tf
    log(f"✅ TensorFlow {tf.__version__} loaded successfully")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import numpy as np
except ImportError as e:
    log(f"❌ Error: {e}")
    log("Please install: pip install tensorflow")
    sys.exit(1)

# Custom callback
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        log(f"\n{'='*70}")
        log(f"📊 EPOCH {epoch + 1}/{self.total_epochs} STARTED")
        log(f"{'='*70}")
        
    def on_epoch_end(self, epoch, logs=None):
        duration = (datetime.now() - self.epoch_start).total_seconds()
        percentage = ((epoch + 1) / self.total_epochs) * 100
        
        log(f"\n{'='*70}")
        log(f"✅ EPOCH {epoch + 1}/{self.total_epochs} COMPLETE - {percentage:.1f}% TOTAL PROGRESS")
        log(f"⏱️  Duration: {duration:.1f} seconds")
        log(f"📈 Train Accuracy: {logs.get('accuracy', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        log(f"📉 Train Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        log(f"{'='*70}")
        
    def on_train_batch_end(self, batch, logs=None):
        # Log every 10 batches
        if batch % 10 == 0:
            log(f"   Batch {batch}: loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}")

# Dataset
dataset_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized\train"
log(f"\n📁 Dataset: {dataset_dir}")

if not os.path.exists(dataset_dir):
    log(f"❌ Dataset not found!")
    sys.exit(1)

# Check classes
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
log(f"📂 Classes found: {len(classes)}")
for cls in classes:
    count = len([f for f in os.listdir(os.path.join(dataset_dir, cls)) if f.endswith(('.jpg', '.jpeg', '.png'))])
    log(f"   - {cls}: {count} images")

log("\n🔧 Creating data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

log(f"✅ Training: {train_gen.samples} samples")
log(f"✅ Validation: {val_gen.samples} samples")

log("\n🏗️  Building model...")

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
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log(f"✅ Model ready - {model.count_params():,} parameters")

os.makedirs('model', exist_ok=True)

epochs = 30  # Reduced for faster training
callbacks = [
    ProgressCallback(epochs),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0),
    ModelCheckpoint('model/copra_model.h5', monitor='val_accuracy', save_best_only=True, verbose=0)
]

log(f"\n🚀 TRAINING START - {epochs} epochs")
log("=" * 70)
log("💡 Monitor this file for real-time progress: training_progress.log")
log("=" * 70)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=0  # Suppress default output, use our callback
)

log("\n" + "=" * 70)
log("🎉 TRAINING COMPLETE!")
log("=" * 70)
log(f"💾 Model: model/copra_model.h5")

with open('model/class_names.txt', 'w') as f:
    for name in train_gen.class_indices.keys():
        f.write(f"{name}\n")
log(f"💾 Classes: model/class_names.txt")
log("=" * 70)
