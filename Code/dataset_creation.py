import mediapipe as mp
import cv2 as cv
import os
import matplotlib.pyplot as plt
import pickle

DATA_DIR = r"D:\Sign Language Recognition\Dataset"

data = []
labels = []


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

for each_dir in os.listdir(DATA_DIR) :
    for each_img in os.listdir(os.path.join(DATA_DIR, each_dir)) :
        data_corr = []
        img = cv.imread(os.path.join(DATA_DIR, each_dir, each_img))
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        results = hands.process(rgb_img)
        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks :
                for i in range(len(hand_landmarks.landmark)) :
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_corr.append(x)
                    data_corr.append(y)

            data.append(data_corr)
            labels.append(each_dir)

our_file = open('dataset.pickle', 'wb')
pickle.dump({
    'data' : data,
    'labels' : labels
}, our_file)
our_file.close()