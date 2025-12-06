"""
Simplified Webcam Demo (No MediaPipe Required)
Purpose: Demonstrate real-time inference without MediaPipe or trained model
Uses: OpenCV's Haar Cascade for face detection + PyTorch model
"""

import cv2
import numpy as np
import torch
import time
from collections import deque

from model import BeautyPredictor


class SimpleBeautyDemo:
    """
    Simplified beauty prediction demo using OpenCV face detection
    
    This demo works without MediaPipe and without a trained ONNX model.
    It uses a randomly initialized PyTorch model for demonstration purposes.
    """
    
    def __init__(self, use_pretrained=True, smooth_window=5):
        """
        Args:
            use_pretrained (bool): Use ImageNet pre-trained weights
            smooth_window (int): Score smoothing window size
        """
        print("Initializing Simplified Beauty Prediction Demo...")
        
        # Load model (PyTorch instead of ONNX)
        print("Loading MobileNetV3 model...")
        self.device = torch.device('cpu')  # CPU only for this demo
        self.model = BeautyPredictor(pretrained=use_pretrained)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded")
        
        # Load Haar Cascade for face detection
        print("Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade!")
        print("✓ Face detector loaded")
        
        # Score smoothing buffer
        self.score_buffer = deque(maxlen=smooth_window)
        
        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print("✓ Demo ready!\n")
    
    def preprocess_face(self, face_img):
        """Preprocess face for model inference"""
        # Resize to 224x224
        img = cv2.resize(face_img, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        # Convert to CHW and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        
        return img_tensor
    
    def predict(self, face_img):
        """Run inference on face image"""
        # Preprocess
        input_tensor = self.preprocess_face(face_img).to(self.device)
        
        # Predict
        with torch.no_grad():
            score = self.model.predict(input_tensor)
        
        return score.cpu().item()
    
    def crop_face_with_padding(self, frame, x, y, w, h, padding=0.2):
        """Crop face with padding"""
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Apply padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame_w, x + w + pad_w)
        y2 = min(frame_h, y + h + pad_h)
        
        face_crop = frame[y1:y2, x1:x2]
        
        return face_crop, (x1, y1, x2, y2)
    
    def get_smoothed_score(self, score):
        """Smooth score using moving average"""
        self.score_buffer.append(score)
        return np.mean(self.score_buffer)
    
    def draw_ui(self, frame, bbox, score, fps):
        """Draw UI overlays"""
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Score text
        score_text = f"Beauty Score: {score:.2f} / 5.00"
        
        # Score background
        (text_w, text_h), _ = cv2.getTextSize(
            score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - 40),
            (x1 + text_w + 10, y1 - 5),
            (0, 255, 0),
            -1
        )
        
        # Draw score
        cv2.putText(
            frame,
            score_text,
            (x1 + 5, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Warning banner
        h, w = frame.shape[:2]
        warning = "DEMO MODE: Untrained model (random predictions)"
        (warn_w, warn_h), _ = cv2.getTextSize(
            warning, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (10, h - 40),
            (10 + warn_w + 10, h - 10),
            (0, 165, 255),
            -1
        )
        cv2.putText(
            frame,
            warning,
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def run(self, camera_id=0):
        """Run the demo"""
        print("="*70)
        print("SIMPLIFIED BEAUTY PREDICTION DEMO")
        print("="*70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset score buffer")
        print("\nNOTE: This demo uses an untrained model.")
        print("      Predictions are for demonstration only!")
        print("="*70 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        print("✓ Camera opened successfully\n")
        
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                
                # Process faces
                if len(faces) > 0:
                    # Take the largest face
                    faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces_sorted[0]
                    
                    # Crop with padding
                    face_crop, bbox = self.crop_face_with_padding(
                        frame, x, y, w, h, padding=0.2
                    )
                    
                    if face_crop.size > 0:
                        try:
                            # Predict
                            score = self.predict(face_crop)
                            smoothed_score = self.get_smoothed_score(score)
                            
                            # Draw UI
                            self.draw_ui(
                                frame,
                                bbox,
                                smoothed_score,
                                np.mean(self.fps_buffer) if self.fps_buffer else 0
                            )
                        except Exception as e:
                            print(f"Prediction error: {e}")
                else:
                    # No face detected
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                
                # Calculate FPS
                frame_time = time.time() - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_buffer.append(fps)
                
                # Show frame
                cv2.imshow('Beauty Prediction Demo', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.score_buffer.clear()
                    print("Score buffer reset")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("Session Summary:")
            print(f"  Total frames: {frame_count}")
            if self.fps_buffer:
                print(f"  Average FPS: {np.mean(self.fps_buffer):.1f}")
            print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Beauty Prediction Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Score smoothing window size')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use ImageNet pre-trained weights')
    
    args = parser.parse_args()
    
    try:
        # Create and run demo
        demo = SimpleBeautyDemo(
            use_pretrained=not args.no_pretrained,
            smooth_window=args.smooth
        )
        demo.run(camera_id=args.camera)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
