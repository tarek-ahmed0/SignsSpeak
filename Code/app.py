# Libraries :
import tkinter as tk  
from tkinter import Text, Label, messagebox 
import numpy as np  
import cv2 as cv  
import mediapipe as mp  
import pickle  
import time  
from PIL import Image, ImageTk, ImageDraw, ImageFont  
import arabic_reshaper  
from bidi.algorithm import get_display  
import pyperclip  
from gtts import gTTS 
import os  

# Load ML Model :
with open('model.pickle', 'rb') as saved_model:  
    loaded_model = pickle.load(saved_model)
    model = loaded_model['model'] 

# Mediapipe Initiation:
mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils  
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)  

# Classes Dictionary :
labels = {
    0: 'السلام عليكم',  
    1: 'كيف الحال',      
    2: 'حذف اخر اشارة'       
}

# Font Configuration :
font_path = r"D:\Sign Language Recognition\Fonts\arial.ttf" 
font_size = 22
font = ImageFont.truetype(font_path, font_size)

# Landmark & Connections Style:
hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)  
hand_connection_style = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1)  

# Main Application Window
root = tk.Tk() 
root.title("Sign Language Recognition")

# Components:
cap = None 
video_label = Label(root)  
text_area = Text(root, height=2, width=50, font=("Arial", 15))
copy_button = tk.Button(root, text="Copy Text", bg="#69359c", fg="white", command=lambda: copy_text())
speech_button = tk.Button(root, text="Download as Voice", bg="#69359c", fg="white", command=lambda: download_voice())

# Variables:
displayed_text_list = []  # List to keep track of displayed texts
displayed_text_for_copy = ""  
last_detected_sign = None  
sign_start_time = None  
confirmation_time = 2.0  
flash_start_time = None  
flash_duration = 0.3  
audio_file = None  

# Start Capture:
def start_creation():
    global cap  
    start_creation_button.pack_forget()  
    video_label.pack()  
    cap = cv.VideoCapture(0) 
    if not cap.isOpened():  
        text_area.insert(tk.END, "Error: Could not open webcam.\n")  
        return

    text_area.pack(pady=10)  
    copy_button.pack(pady=10)  
    speech_button.pack(pady=10)  
    update_frame()

# Frame Update:
def update_frame():
    global displayed_text_list, displayed_text_for_copy, last_detected_sign, sign_start_time, flash_start_time, confirmation_time, flash_duration

    ret, frame = cap.read()  
    if ret: 
        frame = cv.flip(frame, 1)  
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  
        results = hands.process(rgb_frame)  

        predicted_class = None
        predicted_accuracy = 0.0

        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:  
                mp_drawing.draw_landmarks(frame, 
                                          hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS,
                                          hand_landmark_style, hand_connection_style)  
                h, w, _ = frame.shape  
                
                landmark_array = np.array([(landmark.x * w, landmark.y * h) for landmark in hand_landmarks.landmark])
                x_min, y_min = np.min(landmark_array, axis=0).astype(int)  
                x_max, y_max = np.max(landmark_array, axis=0).astype(int)  
                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1) 

                data_corr = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y)]
                
                if len(data_corr) == 42:  
                    predicted_class = int(model.predict([np.asarray(data_corr)])[0])  
                    predicted_accuracy = model.predict_proba([np.asarray(data_corr)])[0][predicted_class]  
                    if predicted_class in labels: 
                        current_sign = labels[predicted_class]  
                        if current_sign != last_detected_sign:  
                            last_detected_sign = current_sign  
                            sign_start_time = time.time()  

                        if sign_start_time and (time.time() - sign_start_time >= confirmation_time):  
                            # Check If The Predicted Class Is 2
                            if predicted_class == 2 :
                                # Remove The Last Predicted Sign
                                if displayed_text_list:
                                    removed_text = displayed_text_list.pop()  
                                    displayed_text_for_copy = ' '.join(displayed_text_list)  
                                    displayed_text = ' '.join(displayed_text_list) 
                            else:
                                displayed_text_list.append(current_sign)  
                                displayed_text_for_copy = ' '.join(displayed_text_list)  
                                flash_start_time = time.time()  
                            sign_start_time = None  

        if flash_start_time and (time.time() - flash_start_time <= flash_duration) and 'x_min' in locals() and 'y_min' in locals():
            overlay = frame.copy()
            alpha = 0.75
            cv.rectangle(overlay, (x_min, y_min), (x_max, y_max), (230, 232, 250), -1)
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            flash_start_time = None 

        # Predicted Class & Accuary Display :
        if predicted_class is not None and predicted_accuracy > 0.5:  
            arabic_text = labels[predicted_class]
            reshaped_text = arabic_reshaper.reshape(arabic_text)  
            bidi_text = get_display(reshaped_text)  
            sign_text = f"{bidi_text} | {(predicted_accuracy)*100}%"

            # PIL Image For Drawing :
            pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Text Dimensions :
            text_bbox = draw.textbbox((0, 0), sign_text, font = font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            rect_x = x_min 
            rect_y = y_min - 12 
            padding = 5 
            draw.rectangle([(rect_x - padding, rect_y - text_height - padding), 
                            (rect_x + text_width + padding, rect_y + padding)], fill = (60, 179, 113))

            text_x = rect_x 
            text_y = rect_y - text_height + padding // 2

            # Draw The Arabic Text :
            draw.text((text_x, text_y), sign_text, font = font, fill = (255, 255, 255))

            # Convert Back To OpenCV Format :
            frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        # Arabic Text Preprocessing :
        reshaped_text_for_display = arabic_reshaper.reshape(' '.join(displayed_text_list))  
        bidi_text_for_display = get_display(reshaped_text_for_display)  

        # Text Area Insertion & Updating :
        text_area.delete("1.0", tk.END)  
        text_area.insert(tk.END, bidi_text_for_display + "\n")

        # Display Frame
        frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=frame_pil)  
        video_label.imgtk = imgtk  
        video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Copy Text :
def copy_text():
    pyperclip.copy(displayed_text_for_copy)  
    messagebox.showinfo("Copied", "Text has been copied to clipboard")  

# Download Voice :
def download_voice():
    if not displayed_text_for_copy.strip():  
        messagebox.showwarning("No Text", "There is no text to convert to voice.")  
        return

    # Google Text To Speech (TTS) :
    tts = gTTS(text=displayed_text_for_copy, lang='ar', slow=False)  
    audio_file = "output.mp3"  
    tts.save(audio_file)  
    os.startfile(audio_file)  
    messagebox.showinfo("Success", f"Audio saved as '{audio_file}'") 

# Start Button
start_creation_button = tk.Button(root, 
                                   text="Start Capturing Signs", 
                                   bg="#69359c", fg="white",
                                   font=("Arial", 16, "bold"),
                                   command=lambda: start_creation())
start_creation_button.pack(pady=50) 

# Main Loop
root.mainloop() 

# Release the video capture on exit:
if cap:  
    cap.release()  
cv.destroyAllWindows()
