import cv2
import mediapipe as mp
import time
import webbrowser

# Constants
FIRST_WEBSITE = "https://en.wikipedia.org/wiki/Stephen_Curry"
SECOND_WEBSITE = "https://en.wikipedia.org/wiki/LeBron_James"
NO_HUMAN_DURATION = 30  # seconds

mp_holistic = mp.solutions.holistic  # Holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
last_seen_human = time.time()

# To track which website is currently open
website_open = None

# Open the first website initially
webbrowser.open(FIRST_WEBSITE)
website_open = "first"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # If a face is detected
    if results.face_landmarks:
        if website_open == "first":
            webbrowser.open(SECOND_WEBSITE)  # Open second website
            website_open = "second"
        last_seen_human = time.time()  # Update the last time a human was seen

    else:
        # If no human detected for NO_HUMAN_DURATION seconds, switch to the first website
        if time.time() - last_seen_human > NO_HUMAN_DURATION and website_open == "second":
            webbrowser.open(FIRST_WEBSITE)
            website_open = "first"

    # For debugging purposes: show the video feed
    annotated_image = frame.copy()
    mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    cv2.imshow('Webcam Feed', annotated_image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
