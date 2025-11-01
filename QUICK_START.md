# 🚀 Quick Start Guide

## ⚡ Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open Your Browser
Go to: `http://127.0.0.1:5000`

### 4. Upload a Coconut Image
- Click "Choose File" or drag & drop an image
- Click "Analyze Purity"
- View results with confidence score

## 🧠 Train Your Own Model

### 1. Prepare Your Dataset
Organize images in folders:
```
dataset/
├── High_Purity/
├── Medium_Purity/
└── Low_Purity/
```

### 2. Run Training
```bash
python train_model.py
```

### 3. Follow Prompts
- Enter dataset path when prompted
- Wait for training to complete
- Model will be saved automatically

## 🔧 Configuration

### File Upload Limits
- **Max Size**: 16MB
- **Supported Formats**: JPG, PNG, GIF, BMP, TIFF

### Model Settings
- **Input Size**: 224x224 pixels
- **Classes**: High, Medium, Low Purity
- **Confidence**: 70-98% typical range

## 🆘 Troubleshooting

### Common Issues

**"TensorFlow not available"**
```bash
pip install tensorflow
```

**"Model not found"**
- Train a model first: `python train_model.py`
- Or use simulation mode (automatic)

**"File too large"**
- Resize image to <16MB
- Use image compression tools

**"Invalid file type"**
- Use supported formats: JPG, PNG, GIF, BMP, TIFF
- Check file extension

### Getting Help
1. Check error messages in terminal
2. Review browser console for JavaScript errors
3. Ensure all dependencies are installed
4. Verify file permissions

## 📱 Mobile Usage

The system is fully responsive and works on:
- 📱 Smartphones
- 📱 Tablets
- 💻 Desktop computers
- 🖥️ Large screens

## 🌐 Cloud Integration

### Google Cloud Vision (Optional)
```bash
pip install google-cloud-vision
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

## 📊 Expected Results

### High Purity (85-98% confidence)
- Clean, fresh appearance
- No visible defects
- Premium quality indicators

### Medium Purity (75-90% confidence)
- Moderate quality
- Some minor imperfections
- Acceptable for general use

### Low Purity (70-85% confidence)
- Visible quality issues
- May require processing
- Consider inspection

## 🎯 Best Practices

### Image Quality
- Use good lighting
- Clear, focused images
- Avoid blurry photos
- Include full coconut in frame

### File Management
- Use descriptive filenames
- Keep file sizes reasonable
- Organize by purity level
- Backup important images

## 🔄 System Updates

### Check for Updates
```bash
git pull origin main
pip install -r requirements.txt
```

### Model Updates
- Retrain with new data
- Validate performance
- Update model file
- Test thoroughly

---

**Need more help? Check the full documentation in `PROJECT_DOCUMENTATION.md`**