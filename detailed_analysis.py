"""
Detailed Beauty Score Analysis with Explanation
Provides comprehensive analysis and improvement suggestions based on facial features
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from model import BeautyPredictor

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
    
    # Skin tone analysis
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
    
    # Sharpness
    sharpness = np.var(laplacian)
    
    return {
        'brightness': float(brightness / 255.0),
        'contrast': float(contrast / 128.0),
        'skin_tone_hue': float(avg_hue / 180.0),
        'skin_saturation': float(avg_saturation / 255.0),
        'skin_brightness': float(avg_value / 255.0),
        'symmetry': float(symmetry_score),
        'smoothness': float(smoothness * 100),
        'sharpness': float(sharpness)
    }


def generate_detailed_explanation(beauty_score, features):
    """
    Generate detailed explanation and improvement suggestions based on score and features
    """
    print("\n" + "="*70)
    print("DETAILED BEAUTY ANALYSIS REPORT")
    print("="*70)
    
    # Score interpretation
    print("\n1. SCORE INTERPRETATION")
    print("-" * 70)
    print(f"   Beauty Score: {beauty_score:.2f} / 5.0")
    
    if beauty_score >= 4.5:
        interpretation = "Exceptional"
        desc = "This score indicates exceptionally attractive facial features."
    elif beauty_score >= 4.0:
        interpretation = "Very Attractive"
        desc = "This score indicates very high attractiveness with well-balanced features."
    elif beauty_score >= 3.5:
        interpretation = "Attractive"
        desc = "This score indicates above-average attractiveness."
    elif beauty_score >= 3.0:
        interpretation = "Above Average"
        desc = "This score indicates good facial features with room for enhancement."
    elif beauty_score >= 2.5:
        interpretation = "Average"
        desc = "This score indicates average facial features."
    else:
        interpretation = "Below Average"
        desc = "This score indicates potential for significant improvement."
    
    print(f"   Category: {interpretation}")
    print(f"   {desc}")
    
    # Strengths analysis
    print("\n2. STRENGTH ANALYSIS")
    print("-" * 70)
    
    strengths = []
    if features['symmetry'] >= 0.85:
        strengths.append(f"   + EXCELLENT Facial Symmetry ({features['symmetry']:.2f}): Your facial features show excellent balance between left and right sides, which is a key factor in perceived beauty.")
    elif features['symmetry'] >= 0.75:
        strengths.append(f"   + GOOD Facial Symmetry ({features['symmetry']:.2f}): Your face shows good symmetry.")
    
    if features['smoothness'] >= 50:
        strengths.append(f"   + EXCELLENT Skin Texture ({features['smoothness']:.1f}): Your skin shows smooth, even texture with minimal irregularities.")
    elif features['smoothness'] >= 30:
        strengths.append(f"   + GOOD Skin Texture ({features['smoothness']:.1f}): Your skin texture is generally smooth.")
    
    if 0.4 <= features['brightness'] <= 0.7:
        strengths.append(f"   + OPTIMAL Lighting/Brightness ({features['brightness']:.2f}): The image has well-balanced lighting that highlights your features.")
    
    if features['contrast'] >= 0.5:
        strengths.append(f"   + DEFINED Features ({features['contrast']:.2f}): Your facial features show good definition and contrast.")
    
    if strengths:
        for strength in strengths:
            print(strength)
    else:
        print("   Analysis shows potential for improvement in multiple areas (see below).")
    
    # Areas for improvement
    print("\n3. AREAS FOR IMPROVEMENT")
    print("-" * 70)
    
    improvements = []
    
    if features['symmetry'] < 0.75:
        improvements.append({
            'area': 'Facial Symmetry',
            'score': features['symmetry'],
            'tips': [
                "Consider professional makeup techniques to balance facial features",
                "Ensure proper photo angles (straight-on shots work best)",
                "Consult a professional stylist for hairstyles that complement your face shape"
            ]
        })
    
    if features['smoothness'] < 30:
        improvements.append({
            'area': 'Skin Texture',
            'score': features['smoothness'],
            'tips': [
                "Establish a consistent skincare routine (cleanse, tone, moisturize)",
                "Use sunscreen daily to protect skin health",
                "Consider professional skincare treatments (facials, chemical peels)",
                "Ensure adequate hydration (8+ glasses of water daily)",
                "Get sufficient sleep (7-9 hours nightly)"
            ]
        })
    
    if features['brightness'] < 0.3 or features['brightness'] > 0.8:
        improvements.append({
            'area': 'Lighting/Photo Quality',
            'score': features['brightness'],
            'tips': [
                "Use natural daylight for photos when possible",
                "Avoid harsh overhead lighting or direct flash",
                "Position yourself facing a window for soft, flattering light",
                "Use ring lights or softbox lighting for even illumination"
            ]
        })
    
    if features['contrast'] < 0.4:
        improvements.append({
            'area': 'Feature Definition',
            'score': features['contrast'],
            'tips': [
                "Use makeup to enhance natural contours (contouring techniques)",
                "Ensure eyebrows are well-groomed and defined",
                "Consider eyelash extensions or mascara to enhance eye definition",
                "Use lip liner and lipstick to define lip shape"
            ]
        })
    
    if features['skin_saturation'] < 0.3:
        improvements.append({
            'area': 'Skin Vitality',
            'score': features['skin_saturation'],
            'tips': [
                "Improve circulation through regular exercise",
                "Use vitamin C serums to brighten skin",
                "Eat a diet rich in fruits and vegetables",
                "Reduce stress through mindfulness or meditation"
            ]
        })
    
    if improvements:
        for imp in improvements:
            print(f"\n   {imp['area']} (Score: {imp['score']:.2f}):")
            for tip in imp['tips']:
                print(f"      - {tip}")
    else:
        print("   Your features are well-balanced! Focus on maintenance:")
        print("      - Continue current skincare routine")
        print("      - Protect skin with daily SPF")
        print("      - Maintain healthy lifestyle habits")
    
    # Personalized recommendations
    print("\n4. PERSONALIZED ACTION PLAN")
    print("-" * 70)
    
    print("\n   IMMEDIATE ACTIONS (Today):")
    print("      1. Review your photo quality and lighting setup")
    print("      2. Start a basic skincare routine if you haven't already")
    print("      3. Ensure you're drinking enough water daily")
    
    print("\n   SHORT-TERM GOALS (This Week):")
    print("      1. Research and invest in quality skincare products")
    print("      2. Practice makeup techniques that enhance your features")
    print("      3. Take new photos with improved lighting")
    print("      4. Get adequate sleep (7-9 hours nightly)")
    
    print("\n   LONG-TERM IMPROVEMENTS (This Month+):")
    print("      1. Establish consistent skincare and grooming routines")
    print("      2. Consider professional consultations (dermatologist, stylist)")
    print("      3. Monitor diet and exercise for overall health")
    print("      4. Re-test in 30 days to track improvements")
    
    print("\n5. TECHNICAL METRICS SUMMARY")
    print("-" * 70)
    for key, value in features.items():
        status = ""
        if key == 'symmetry':
            status = "Excellent" if value >= 0.85 else "Good" if value >= 0.75 else "Needs Work"
        elif key == 'smoothness':
            status = "Excellent" if value >= 50 else "Good" if value >= 30 else "Needs Work"
        elif key == 'brightness':
            status = "Optimal" if 0.4 <= value <= 0.7 else "Adjust Lighting"
        elif key == 'contrast':
            status = "Good" if value >= 0.5 else "Needs Enhancement"
        
        print(f"   {key.replace('_', ' ').title()}: {value:.2f} - {status}")


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
        return None, None
    
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
        return None, None
    
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
    
    return beauty_score, face_crop


def main():
    image_path = r'C:\Users\DELL\Downloads\Gemini_Generated_Image_srkzensrkzensrkz.png'
    
    print("="*70)
    print("COMPREHENSIVE BEAUTY ANALYSIS SYSTEM")
    print("="*70)
    
    # Get prediction
    beauty_score, face_crop = load_model_and_predict(image_path)
    
    if beauty_score is None:
        return
    
    # Analyze facial features
    print("\nAnalyzing facial features...")
    features = analyze_facial_features(face_crop)
    
    # Generate detailed explanation
    generate_detailed_explanation(beauty_score, features)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNote: This analysis is based on AI predictions and computer vision")
    print("metrics. Beauty is subjective and multifaceted. These suggestions")
    print("are meant to be helpful, not prescriptive. Your unique features")
    print("are what make you special!")
    print("="*70)


if __name__ == "__main__":
    main()
