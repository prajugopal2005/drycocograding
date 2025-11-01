"""
Improved training script with better settings to prevent overfitting
- Simpler model architecture
- Class weights for balanced training
- Better data augmentation
- Higher validation split
"""

import os
import sys
from datetime import datetime
import numpy as np

# Setup logging
log_file = 'training_progress_improved.log'

def log(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

# Clear previous log
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('')

log("=" * 70)
log("🥥 IMPROVED COPRA GRADING MODEL TRAINING")
log("=" * 70)

# Load TensorFlow
log("📦 Loading TensorFlow...")
try:
    import tensorflow as tf
    log(f"✅ TensorFlow {tf.__version__} loaded")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    log(f"❌ Error: {e}")
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

# Dataset
dataset_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized\train"
log(f"\n📁 Dataset: {dataset_dir}")

if not os.path.exists(dataset_dir):
    log(f"❌ Dataset not found!")
    sys.exit(1)

# Check classes
classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
log(f"📂 Classes found: {len(classes)}")
for cls in classes:
    count = len([f for f in os.listdir(os.path.join(dataset_dir, cls)) if f.endswith(('.jpg', '.jpeg', '.png'))])
    log(f"   - {cls}: {count} images")

log("\n🔧 Creating data generators with improved augmentation...")

# More aggressive data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.3  # Increased validation split
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,  # Smaller batch size
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

log(f"✅ Training: {train_gen.samples} samples")
log(f"✅ Validation: {val_gen.samples} samples")

# Calculate class weights for balanced training
log("\n⚖️  Calculating class weights...")
class_indices = train_gen.class_indices
class_labels = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights = dict(enumerate(class_weights_array))
log(f"✅ Class weights: {class_weights}")

log("\n🏗️  Building SIMPLER model to prevent overfitting...")

# Simpler model architecture
model = Sequential([
    # First block
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    
    # Second block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    
    # Third block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    
    # Global pooling instead of flatten
    GlobalAveragePooling2D(),
    
    # Dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log(f"✅ Model ready - {model.count_params():,} parameters")

os.makedirs('model', exist_ok=True)

epochs = 50
callbacks = [
    ProgressCallback(epochs),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=0, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0),
    ModelCheckpoint('model/copra_model_improved.h5', monitor='val_accuracy', save_best_only=True, verbose=0, mode='max')
]

log(f"\n🚀 TRAINING START - {epochs} epochs")
log("=" * 70)
log("💡 Monitor: training_progress_improved.log")
log("⚖️  Using class weights for balanced training")
log("=" * 70)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights,  # Apply class weights
    verbose=0
)

log("\n" + "=" * 70)
log("🎉 TRAINING COMPLETE!")
log("=" * 70)
log(f"💾 Model: model/copra_model_improved.h5")

# Save class names
with open('model/class_names.txt', 'w') as f:
    for name in sorted(train_gen.class_indices.keys()):
        f.write(f"{name}\n")
log(f"💾 Classes: model/class_names.txt")

# Print best results
best_epoch = np.argmax(history.history['val_accuracy'])
log(f"\n📊 Best Epoch: {best_epoch + 1}")
log(f"   Train Accuracy: {history.history['accuracy'][best_epoch]:.4f}")
log(f"   Val Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
log("=" * 70)
