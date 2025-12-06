"""
AI-Powered Beauty Score Explanation System
Uses Groq LLM to analyze facial features and provide personalized suggestions
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from model import BeautyPredictor
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Groq client
client = OpenAI(
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.groq.com/openai/v1'),
    api_key=os.getenv('GROQ_API_KEY')
)
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def analyze_facial_features(face_image):
    """
    Analyze facial features using computer vision
    Returns structured data about facial characteristics
    """
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    # Skin tone analysis (V channel represents brightness/value)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # Symmetry analysis (compare left and right halves)
    height, width = gray.shape
    left_half = gray[:, :width//2]
    right_half = cv2.flip(gray[:, width//2:], 1)
    
    # Resize to match if needed
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = right_half[:, :min_width]
    
    symmetry_score = 1.0 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0)
    
    # Texture analysis (smoothness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    smoothness = 1.0 / (1.0 + np.var(laplacian))
    
    return {
        'brightness': float(brightness / 255.0),
        'contrast': float(contrast / 128.0),
        'skin_tone_hue': float(avg_hue / 180.0),
        'skin_saturation': float(avg_saturation / 255.0),
        'skin_brightness': float(avg_value / 255.0),
        'symmetry': float(symmetry_score),
        'smoothness': float(smoothness * 100)
    }


def get_ai_explanation(beauty_score, features):
    """
    Use Groq LLM to generate personalized explanation and improvement suggestions
    """
    # Create a detailed prompt for the LLM
    prompt = f"""You are an expert beauty consultant and facial aesthetics specialist. Analyze this beauty prediction result and provide professional, constructive feedback.

BEAUTY SCORE: {beauty_score:.2f} / 5.0

FACIAL ANALYSIS METRICS:
- Brightness: {features['brightness']:.2f} (0.0 = dark, 1.0 = bright)
- Contrast: {features['contrast']:.2f} (higher = more defined features)
- Skin Tone Hue: {features['skin_tone_hue']:.2f}
- Skin Saturation: {features['skin_saturation']:.2f} (0.0 = pale, 1.0 = vibrant)
- Skin Brightness: {features['skin_brightness']:.2f}
- Facial Symmetry: {features['symmetry']:.2f} (0.0 = asymmetric, 1.0 = perfectly symmetric)
- Skin Smoothness: {features['smoothness']:.2f} (higher = smoother)

Please provide:

1. **SCORE INTERPRETATION** (2-3 sentences)
   - What does this score mean in terms of attractiveness?
   - Overall impression based on the metrics

2. **STRENGTHS** (2-3 specific points)
   - What facial features are contributing positively to the score?
   - Which metrics show strong performance?

3. **AREAS FOR IMPROVEMENT** (3-5 actionable suggestions)
   - Specific, practical tips for enhancing appearance
   - Based on the weaker metrics
   - Include skincare, grooming, photography, and styling suggestions

4. **PERSONALIZED RECOMMENDATIONS** (2-3 specific actions)
   - Immediate steps they can take
   - Long-term beauty routine suggestions

Be professional, supportive, and constructive. Focus on actionable advice rather than criticism."""

    try:
        # Call Groq API
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional beauty consultant providing constructive, supportive, and actionable feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating AI explanation: {str(e)}\n\nPlease check your API credentials."


def load_model_and_predict(image_path):
    """Load model and make prediction"""
    print("Loading model...")
    device = torch.device('cpu')
    model = BeautyPredictor(pretrained=False)
    
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("+ Model loaded successfully")
    else:
        print("X Model checkpoint not found!")
        return None, None, None
    
    model.to(device)
    model.eval()
    
    # Load face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("X Failed to load image!")
        return None, None, None
    
    print(f"+ Image loaded: {image.shape}")
    
    # Detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        print("X No face detected in image!")
        print("   Using entire image instead...")
        face_crop = image
    else:
        print(f"+ Found {len(faces)} face(s)")
        faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        
        frame_h, frame_w = image.shape[:2]
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame_w, x + w + pad_w)
        y2 = min(frame_h, y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        print(f"+ Face cropped: {face_crop.shape}")
    
    # Preprocess for model
    img = cv2.resize(face_crop, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - MEAN) / STD
    img_tensor_data = np.transpose(img_norm, (2, 0, 1))
    img_tensor_data = np.expand_dims(img_tensor_data, axis=0)
    img_tensor = torch.from_numpy(img_tensor_data).float().to(device)
    
    # Predict
    print("\nMaking prediction...")
    with torch.no_grad():
        score = model.predict(img_tensor)
    
    beauty_score = float(score.cpu().item())
    
    return beauty_score, face_crop, image


def main():
    # Image path
    image_path = r'C:\Users\DELL\Downloads\Gemini_Generated_Image_srkzensrkzensrkz.png'
    
    print("="*70)
    print("AI-POWERED BEAUTY ANALYSIS SYSTEM")
    print("="*70)
    
    # Get prediction
    beauty_score, face_crop, original_image = load_model_and_predict(image_path)
    
    if beauty_score is None:
        return
    
    # Analyze facial features
    print("\nAnalyzing facial features...")
    features = analyze_facial_features(face_crop)
    
    print("\n" + "="*70)
    print(f"BEAUTY SCORE: {beauty_score:.2f} / 5.0")
    print("="*70)
    
    print("\nFACIAL METRICS:")
    for key, value in features.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    
    # Get AI explanation
    print("\n" + "="*70)
    print("GENERATING AI ANALYSIS...")
    print("="*70)
    
    explanation = get_ai_explanation(beauty_score, features)
    
    print("\n" + explanation)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
