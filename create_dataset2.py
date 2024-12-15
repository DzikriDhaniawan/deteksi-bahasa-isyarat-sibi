import os

import mediapipe as mp
import cv2
import matplotlib .pyplot as plt


mp_hands =mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,min_detection_confidence=0.3)

DATA_DIR= './data'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    print(hand_landmarks.landmark[i])

        plt.figure()
        plt.imshow(img_rgb)

plt.show()