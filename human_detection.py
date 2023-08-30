import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # If a face is detected
    if results.face_landmarks:
        print("Human Detected")
    else:
        print("No Human Detected")

    # For debugging purposes: show the video feed
    annotated_image = frame.copy()
    mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    cv2.imshow('Webcam Feed', annotated_image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()