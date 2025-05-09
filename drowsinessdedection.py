import cv2
import mediapipe as mp
import numpy as np
import time
import pygame


pygame.init()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.15
DROWSY_FRAMES_THRESHOLD = 30
drowsy_frames = 0
alert_active = False
last_alert_time = 0
ALERT_COOLDOWN = 3.0

def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR)
    EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)
    """
    p1 = eye_landmarks[0]
    p2 = eye_landmarks[1]
    p3 = eye_landmarks[2]
    p4 = eye_landmarks[3]
    p5 = eye_landmarks[4]
    p6 = eye_landmarks[5]
    
    distance_p2p6 = np.linalg.norm(np.array(p2) - np.array(p6))
    distance_p3p5 = np.linalg.norm(np.array(p3) - np.array(p5))
    distance_p1p4 = np.linalg.norm(np.array(p1) - np.array(p4))
    
    ear = (distance_p2p6 + distance_p3p5) / (2.0 * distance_p1p4)
    return ear

def main():
    global drowsy_frames, alert_active, last_alert_time
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape
                
                left_eye_coords = []
                right_eye_coords = []
                
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    left_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    right_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                left_ear = calculate_ear(left_eye_coords)
                right_ear = calculate_ear(right_eye_coords)
                
                ear = (left_ear + right_ear) / 2.0
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if ear < EAR_THRESHOLD:
                    drowsy_frames += 1
                    
                    if drowsy_frames >= DROWSY_FRAMES_THRESHOLD:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        current_time = time.time()
                        if not alert_active or (current_time - last_alert_time) > ALERT_COOLDOWN:
                            alert_sound.play()
                            alert_active = True
                            last_alert_time = current_time
                else:
                    drowsy_frames = 0
                    alert_active = False
        
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()