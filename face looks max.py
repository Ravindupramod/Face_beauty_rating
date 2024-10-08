import cv2
import numpy as np
import time

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_smile.xml')
eye_glass_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_eye_tree_eyeglasses.xml')
profile_face_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_profileface.xml')
mouth_cascade = cv2.CascadeClassifier(r'F:\Chrome downloads\haarcascade_mcs_mouth.xml')

def calculate_feature_ratings(eyes_detected, smile_detected, mouth_detected, profile_detected):
    """Calculate ratings for eyes, smile, mouth, and profile."""
    eye_rating = min(eyes_detected * 50, 100)  # Each eye detected contributes 50 points, max 100
    smile_rating = 50 if smile_detected else 0  # 50 points if smile is detected
    mouth_rating = 50 if mouth_detected else 0  # 50 points if mouth is detected
    profile_rating = 20 if profile_detected else 0  # 20 points if profile is detected

    return eye_rating, smile_rating, mouth_rating, profile_rating

def calculate_beauty_score(frame, face_coordinates, eyes_detected, smile_detected, mouth_detected, profile_detected):
    """Calculate a beauty score based on face metrics."""
    x, y, w, h = face_coordinates
    
    # Symmetry score based on width/height ratio
    symmetry_score = min(1.0, (w / h) * 2) * 100
    
    # Get ratings for individual features
    eye_rating, smile_rating, mouth_rating, profile_rating = calculate_feature_ratings(eyes_detected, smile_detected, mouth_detected, profile_detected)
    
    # Brightness score
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0])
    brightness_score = min(brightness / 255, 1) * 100  
    
    # Final beauty score calculation
    beauty_score = (symmetry_score + eye_rating + smile_rating + mouth_rating + profile_rating + brightness_score) / 6  
    
    return beauty_score, eye_rating, smile_rating, mouth_rating, profile_rating

def enhance_features(image, face_coordinates):
    """Enhance facial features for looks maximization."""
    x, y, w, h = face_coordinates
    face_region = image[y:y + h, x:x + w]

    # Convert to YUV color space to manipulate brightness
    yuv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2YUV)
    yuv_face[:, :, 0] = cv2.add(yuv_face[:, :, 0], 30)  # Increase brightness

    # Convert back to BGR
    enhanced_face = cv2.cvtColor(yuv_face, cv2.COLOR_YUV2BGR)

    # Smooth skin using Gaussian Blur
    smooth_face = cv2.GaussianBlur(enhanced_face, (15, 15), 30)

    # Detect eyes, smiles, and mouth
    gray_face = cv2.cvtColor(smooth_face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    eye_glasses = eye_glass_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)
    mouths = mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=20)

    # Brighten eyes
    for (ex, ey, ew, eh) in eyes:
        eye_region = smooth_face[ey:ey + eh, ex:ex + ew]
        eye_region = cv2.add(eye_region, (50, 50, 50))
        smooth_face[ey:ey + eh, ex:ex + ew] = eye_region

    return smooth_face, len(eyes), len(smiles) > 0, len(mouths) > 0, len(eye_glasses) > 0

def detect_and_enhance_faces(duration=30):
    cap = cv2.VideoCapture(0)  # Capture from webcam
    start_time = time.time()
    
    beauty_scores = []
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Enhance the facial features
            enhanced_face, eyes_detected, smile_detected, mouth_detected, profile_detected = enhance_features(frame, (x, y, w, h))

            # Replace the original face region with the enhanced one
            frame[y:y + h, x:x + w] = enhanced_face

            # Calculate and store beauty score and feature ratings
            beauty_score, eye_rating, smile_rating, mouth_rating, profile_rating = calculate_beauty_score(frame, (x, y, w, h), eyes_detected, smile_detected, mouth_detected, profile_detected)
            beauty_scores.append(beauty_score)
            face_count += 1
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the frame with enhanced face
            cv2.imwrite('enhanced_face_snapshot.jpg', frame)

            # Display ratings on the frame
            cv2.putText(frame, f"Beauty Score: {beauty_score:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Rating: {eye_rating:.2f}%", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Smile Rating: {smile_rating:.2f}%", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f"Mouth Rating: {mouth_rating:.2f}%", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Profile Rating: {profile_rating:.2f}%", (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the output frame
        cv2.imshow('Looks Maximization', frame)

        # Check if the duration has been reached
        if time.time() - start_time > duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Summary Report
    if face_count > 0:
        average_beauty_score = sum(beauty_scores) / face_count
    else:
        average_beauty_score = 0

    print(f"\nSummary Report:")
    print(f"-------------------")
    print(f"Total Faces Detected: {face_count}")
    print(f"Average Beauty Score: {average_beauty_score:.2f}%")

    # Ask user to exit or retake
    while True:
        user_choice = input("\nDo you want to (E)xit or (R)etake the assessment? ").strip().upper()
        if user_choice == 'E':
            print("Exiting the program. Thank you!")
            break
        elif user_choice == 'R':
            print("Retaking the assessment...")
            detect_and_enhance_faces(duration=30)
            break
        else:
            print("Invalid input. Please enter 'E' to exit or 'R' to retake.")

# Run the face detection and enhancement function for 30 seconds
if __name__ == '__main__':
    detect_and_enhance_faces(duration=30)
