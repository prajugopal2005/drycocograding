"""
Model Training Script for Coconut Purity Grading System
Trains a CNN model on user-provided dataset for dry coconut purity classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Please install: pip install tensorflow")

def create_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Create a CNN model for coconut purity classification
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of purity classes (High, Medium, Low)
        
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for model training")
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(data_dir, img_size=(224, 224), test_size=0.2, validation_size=0.2):
    """
    Prepare dataset for training
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (tuple): Target image size
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of training data for validation
        
    Returns:
        tuple: (train_gen, val_gen, test_gen, class_names)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for data preparation")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=validation_size
    )
    
    # No augmentation for validation and test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names

def train_model(data_dir, epochs=50, img_size=(224, 224)):
    """
    Train the coconut purity classification model
    
    Args:
        data_dir (str): Path to dataset directory
        epochs (int): Number of training epochs
        img_size (tuple): Target image size
        
    Returns:
        tuple: (trained_model, history, class_names)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for model training")
    
    print("=" * 60)
    print("🥥 Training Coconut Purity Classification Model")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Prepare data
    print("📊 Preparing dataset...")
    train_gen, val_gen, class_names = prepare_data(data_dir, img_size)
    
    print(f"📁 Found {len(class_names)} classes: {class_names}")
    print(f"📈 Training samples: {train_gen.samples}")
    print(f"📈 Validation samples: {val_gen.samples}")
    
    # Create model
    print("🏗️ Creating model architecture...")
    model = create_model(input_shape=(*img_size, 3), num_classes=len(class_names))
    
    # Print model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint('model/coconut_purity_model.h5', monitor='val_accuracy', 
                       save_best_only=True, save_weights_only=False)
    ]
    
    # Train the model
    print(f"\n🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed!")
    return model, history, class_names

def plot_training_history(history):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Keras training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_data_dir, class_names, img_size=(224, 224)):
    """
    Evaluate the trained model
    
    Args:
        model: Trained Keras model
        test_data_dir (str): Path to test dataset
        class_names (list): List of class names
        img_size (tuple): Target image size
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for model evaluation")
    
    print("\n📊 Evaluating model...")
    
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function
    """
    import sys
    
    print("🥥 Coconut Purity Grading - Model Training")
    print("=" * 50)
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow is not installed!")
        print("Please install with: pip install tensorflow")
        return
    
    # Dataset directory (from command line or user input)
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1].strip()
    else:
        dataset_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized\train"
        print(f"📁 Using default dataset directory: {dataset_dir}")
    
    if not dataset_dir:
        print("❌ No dataset directory provided!")
        print("Please organize your dataset as follows:")
        print("dataset/")
        print("├── High_Purity/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── Medium_Purity/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("└── Low_Purity/")
        print("    ├── image1.jpg")
        print("    └── image2.jpg")
        return
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    try:
        # Create model directory
        os.makedirs('model', exist_ok=True)
        
        # Train the model
        model, history, class_names = train_model(dataset_dir, epochs=50)
        
        # Plot training history
        plot_training_history(history)
        
        # Save model
        model.save('model/coconut_purity_model.h5')
        print("💾 Model saved to: model/coconut_purity_model.h5")
        
        # Save class names
        with open('model/class_names.txt', 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        print("💾 Class names saved to: model/class_names.txt")
        
        print("\n🎉 Training completed successfully!")
        print("You can now use the trained model in your Flask application.")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main()
