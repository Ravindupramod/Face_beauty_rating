"""
Real-Time Facial Beauty Prediction Application
Author: Senior Computer Vision Engineer
Purpose: Webcam-based beauty prediction with <150ms latency using ONNX Runtime
"""

import cv2
import numpy as np
import time
from collections import deque
import argparse

import mediapipe as mp
import onnxruntime as ort


class BeautyPredictionApp:
    """
    Real-time facial beauty prediction application
    
    Features:
    - MediaPipe face detection (<5ms latency)
    - ONNX Runtime CPU inference
    - Score smoothing to prevent flickering
    - Ethical disclaimer overlay
    """
    
    def __init__(
        self,
        model_path='beauty_model_int8.onnx',
        smooth_window=5,
        confidence_threshold=0.7
    ):
        """
        Args:
            model_path (str): Path to quantized ONNX model
            smooth_window (int): Number of frames for score smoothing
            confidence_threshold (float): Min face detection confidence
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize ONNX Runtime session
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Inference provider: {self.session.get_providers()}")
        
        # Initialize MediaPipe Face Detection
        print("Initializing MediaPipe Face Detection...")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (faster), 1 = full-range
            min_detection_confidence=confidence_threshold
        )
        print("✓ MediaPipe initialized")
        
        # Score smoothing buffer
        self.score_buffer = deque(maxlen=smooth_window)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model inference
        Same pipeline as training: Resize -> Normalize
        
        Args:
            face_img (ndarray): Face crop in BGR format
        
        Returns:
            ndarray: Preprocessed image [1, 3, 224, 224]
        """
        # Resize to 224x224
        img = cv2.resize(face_img, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        # Convert to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, face_img):
        """
        Run inference on face image
        
        Args:
            face_img (ndarray): Face crop in BGR format
        
        Returns:
            float: Predicted beauty score (1.0 - 5.0)
        """
        # Preprocess
        input_tensor = self.preprocess_face(face_img)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        
        # Get score and clamp to valid range
        score = float(outputs[0][0][0])
        score = np.clip(score, 1.0, 5.0)
        
        return score
    
    def crop_face_with_padding(self, frame, detection):
        """
        Crop face with 20% context padding
        Same as training pipeline
        
        Args:
            frame (ndarray): Full frame image
            detection: MediaPipe detection object
        
        Returns:
            ndarray: Cropped face with padding
            tuple: Bounding box coordinates (x1, y1, x2, y2)
        """
        h, w, _ = frame.shape
        
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        # Apply 20% padding
        padding_w = int(box_w * 0.2)
        padding_h = int(box_h * 0.2)
        
        # Calculate padded coordinates
        x1_pad = max(0, x1 - padding_w)
        y1_pad = max(0, y1 - padding_h)
        x2_pad = min(w, x1 + box_w + padding_w)
        y2_pad = min(h, y1 + box_h + padding_h)
        
        # Crop face
        face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        return face_crop, (x1_pad, y1_pad, x2_pad, y2_pad)
    
    def get_smoothed_score(self, score):
        """
        Smooth score using moving average to prevent flickering
        
        Args:
            score (float): Current frame score
        
        Returns:
            float: Smoothed score
        """
        self.score_buffer.append(score)
        return np.mean(self.score_buffer)
    
    def draw_ui(self, frame, bbox, score, confidence, fps):
        """
        Draw UI overlays on frame
        
        Args:
            frame (ndarray): Video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            score (float): Beauty score
            confidence (float): Detection confidence
            fps (float): Current FPS
        """
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw score background
        score_text = f"Score: {score:.2f} / 5.00"
        conf_text = f"Conf: {confidence:.2f}"
        
        # Score box
        (text_w, text_h), _ = cv2.getTextSize(
            score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - 60),
            (x1 + text_w + 10, y1 - 10),
            (0, 255, 0),
            -1
        )
        
        # Draw score
        cv2.putText(
            frame,
            score_text,
            (x1 + 5, y1 - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Draw confidence
        cv2.putText(
            frame,
            conf_text,
            (x1 + 5, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Draw disclaimer (MANDATORY)
        disclaimer = "Est. based on SCUT-FBP5500 (Asian/White data only)"
        h, w = frame.shape[:2]
        
        # Disclaimer background
        (disc_w, disc_h), _ = cv2.getTextSize(
            disclaimer, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (10, h - 40),
            (10 + disc_w + 10, h - 10),
            (0, 0, 255),
            -1
        )
        
        # Disclaimer text
        cv2.putText(
            frame,
            disclaimer,
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def run(self, camera_id=0):
        """
        Run the real-time application
        
        Args:
            camera_id (int): Camera device ID
        """
        print(f"\n{'='*60}")
        print("Starting Real-Time Beauty Prediction")
        print(f"{'='*60}\n")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset score buffer")
        print(f"\n{'='*60}\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        print(f"✓ Camera opened successfully")
        
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = self.face_detection.process(rgb_frame)
                
                # Process detections
                if results.detections:
                    for detection in results.detections:
                        # Get confidence
                        confidence = detection.score[0]
                        
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Crop face with padding
                        face_crop, bbox = self.crop_face_with_padding(
                            frame, detection
                        )
                        
                        if face_crop.size == 0:
                            continue
                        
                        # Predict beauty score
                        try:
                            score = self.predict(face_crop)
                            smoothed_score = self.get_smoothed_score(score)
                            
                            # Draw UI
                            self.draw_ui(
                                frame,
                                bbox,
                                smoothed_score,
                                confidence,
                                np.mean(self.fps_buffer) if self.fps_buffer else 0
                            )
                            
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            continue
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
                cv2.imshow('Beauty Prediction', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.score_buffer.clear()
                    print("Score buffer reset")
                
                frame_count += 1
                
                # Print latency warning if needed
                if frame_time * 1000 > 150:
                    print(f"⚠ Warning: Frame latency {frame_time*1000:.1f}ms exceeds 150ms target")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.face_detection.close()
            
            print(f"\n{'='*60}")
            print("Session Summary:")
            print(f"  Total frames: {frame_count}")
            if self.fps_buffer:
                print(f"  Average FPS: {np.mean(self.fps_buffer):.1f}")
                avg_latency = 1000 / np.mean(self.fps_buffer)
                print(f"  Average latency: {avg_latency:.1f}ms")
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Real-Time Beauty Prediction')
    parser.add_argument('--model', type=str, default='beauty_model_int8.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Score smoothing window size')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Minimum face detection confidence')
    
    args = parser.parse_args()
    
    # Create app
    app = BeautyPredictionApp(
        model_path=args.model,
        smooth_window=args.smooth,
        confidence_threshold=args.confidence
    )
    
    # Run
    app.run(camera_id=args.camera)


if __name__ == '__main__':
    main()
