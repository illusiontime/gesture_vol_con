import cv2
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np
from comtypes import CoInitialize

# Initialize COM
CoInitialize()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
except Exception as e:
    print(f"Error initializing Pycaw: {e}")
    exit()

vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]
smoothed_vol = min_vol

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw points on thumb and index finger tips
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

            # Calculate distance between thumb and index finger
            try:
                distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y)** 2)
                distance = np.clip(distance, 20, 200)  # Clamp distance
            except ValueError as e:
                print(f"Error calculating distance: {e}")
                exit()

            # Map the distance to volume range
            vol = np.interp(distance, [20, 200], [min_vol, max_vol])
            smoothed_vol = 0.9 * smoothed_vol + 0.1 * vol  # Smooth the volume
            volume.SetMasterVolumeLevel(smoothed_vol, None)

            # Display the distance on screen
            cv2.putText(frame, f'Volume: {int(np.interp(smoothed_vol, [min_vol, max_vol], [0, 100]))}%', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Gesture Volume Control', frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()