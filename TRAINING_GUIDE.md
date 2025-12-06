# 🎯 Complete Project Setup & Training Guide

## Quick Start - 3 Steps to Complete Beauty Prediction

### Step 1: Download SCUT-FBP5500 Dataset

**Option A: Direct Download (Recommended)**
1. Open this link in your browser: https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypqrvfGwf
2. Click "Download" (172 MB)
3. Save to project directory
4. Extract the ZIP file

**Option B: Alternative Sources**
- Baidu Pan (China): https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw (Password: `if7p`)
- Hugging Face: https://huggingface.co/datasets/SCUT-FBP5500

**Expected Folder Structure After Extraction:**
```
SCUT-FBP5500/
├── Images/
│   ├── AF1.jpg
│   ├── AF2.jpg
│   └── ... (5500 images total)
└── train_test_files/
    ├── All_Ratings.xlsx
    └── split_of_60%training and 40%testing/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

---

### Step 2: Train the Model

Once the dataset is downloaded and extracted:

```bash
# Start training (will take 2-4 hours on CPU)
python train.py --data_dir ./SCUT-FBP5500 --epochs 50 --batch_size 32

# Or with fewer epochs for faster testing
python train.py --data_dir ./SCUT-FBP5500 --epochs 10 --batch_size 16
```

**Training Progress:**
- Epoch 1-10: Model learns basic patterns (~0.60 Pearson)
- Epoch 10-30: Performance improves (~0.75-0.85 Pearson)
- Epoch 30-50: Fine-tuning (~0.88-0.92 Pearson)

**What Happens During Training:**
1. ✅ Loads SCUT-FBP5500 dataset
2. ✅ Trains MobileNetV3-Large model
3. ✅ Saves best model to `checkpoints/best_model.pth`
4. ✅ Automatically exports to `beauty_model.onnx`
5. ✅ Applies Int8 quantization → `beauty_model_int8.onnx`

---

### Step 3: Test Your Trained Model

#### Option A: Web UI
```bash
# Start API server with trained model
python api.py

# Open browser to http://localhost:8000
# Upload your photo and get real predictions!
```

#### Option B: Webcam
```bash
# Real-time predictions with webcam
python demo_webcam.py
```

#### Option C: Command Line
```bash
# Single image prediction
curl -X POST -F "file=@your_photo.jpg" http://localhost:8000/predict
```

---

## 📊 Expected Training Results

| Metric | Target | Typical Result |
|--------|--------|----------------|
| Pearson Correlation | >0.85 | 0.88 - 0.92 |
| MAE | <0.30 | 0.20 - 0.25 |
| RMSE | <0.35 | 0.25 - 0.30 |
| Training Time | - | 2-4 hours (CPU) |
| Model Size (Int8) | <5MB | 3-4 MB |

---

## 🚀 Deployment After Training

### Local Deployment
```bash
# API is already configured
python api.py
# Access at http://localhost:8000
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Access at http://localhost
```

### Cloud Deployment
See [DEPLOYMENT.md](file:///d:/AI-Generated%20Personalized%20Smell%20Profiles/AI-Generated%20Personalized%20Smell%20Profiles/DEPLOYMENT.md) for AWS, GCP, Azure guides.

---

## 🔧 Troubleshooting

### Dataset Not Found
```bash
# Check if dataset exists
ls SCUT-FBP5500/Images/

# If not, download manually from Google Drive
```

### Out of Memory During Training
```bash
# Reduce batch size
python train.py --data_dir ./SCUT-FBP5500 --batch_size 8 --epochs 50
```

### Training Too Slow
```bash
# Use fewer epochs for testing
python train.py --data_dir ./SCUT-FBP5500 --epochs 10

# Or use GPU if available
python train.py --data_dir ./SCUT-FBP5500 --device cuda
```

---

## ✅ Project Completion Checklist

### Before Training
- [ ] Download SCUT-FBP5500 dataset (172 MB)
- [ ] Extract to project directory
- [ ] Verify folder structure
- [ ] Install dependencies: `pip install -r requirements-prod.txt`

### Training
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Wait for completion (2-4 hours)
- [ ] Verify best model saved in `checkpoints/`

### After Training
- [ ] ONNX model exported: `beauty_model.onnx`
- [ ] Quantized model created: `beauty_model_int8.onnx`
- [ ] Test with web UI
- [ ] Test with webcam
- [ ] Verify Pearson correlation >0.85

### Deployment
- [ ] API server tested locally
- [ ] Docker image built (optional)
- [ ] Deploy to production (optional)
- [ ] Configure SSL/HTTPS (production only)

---

## 🎉 Project Complete When:

1. ✅ **Dataset Downloaded** - SCUT-FBP5500 in project folder
2. ✅ **Model Trained** - Best model saved with >0.85 Pearson correlation
3. ✅ **ONNX Exported** - Quantized model <5MB
4. ✅ **Testing Works** - Web UI shows accurate predictions
5. ✅ **Deployed** - API accessible via browser

---

## 📝 Quick Reference Commands

```bash
# 1. Download dataset manually from browser
# https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypqrvfGwf

# 2. Train model
python train.py --data_dir ./SCUT-FBP5500 --epochs 50

# 3. Start API
python api.py

# 4. Open browser
# http://localhost:8000
```

---

## 💡 Pro Tips

1. **Quick Test**: Train with 10 epochs first to verify everything works
2. **GPU Training**: Much faster if you have CUDA-capable GPU
3. **Model Export**: Happens automatically after training
4. **Backup**: Save `checkpoints/best_model.pth` - it's your trained model!
5. **Production**: Use the quantized Int8 model for deployment

---

## 📚 Additional Resources

- **Training Script**: [train.py](file:///d:/AI-Generated%20Personalized%20Smell%20Profiles/AI-Generated%20Personalized%20Smell%20Profiles/train.py)
- **API Service**: [api.py](file:///d:/AI-Generated%20Personalized%20Smell%20Profiles/AI-Generated%20Personalized%20Smell%20Profiles/api.py)
- **Deployment Guide**: [DEPLOYMENT.md](file:///d:/AI-Generated%20Personalized%20Smell%20Profiles/AI-Generated%20Personalized%20Smell%20Profiles/DEPLOYMENT.md)
- **Project README**: [README.md](file:///d:/AI-Generated%20Personalized%20Smell%20Profiles/AI-Generated%20Personalized%20Smell%20Profiles/README.md)

---

**Total Time to Complete: 3-5 hours** (mostly training time)

The project is **99% complete** - just needs the dataset and training! 🚀
