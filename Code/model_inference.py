import numpy as np
import cv2 as cv
import mediapipe as mp
import pickle
import time
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# Load the pre-trained model
with open('model.pickle', 'rb') as saved_model:
    loaded_model = pickle.load(saved_model)
    model = loaded_model['model']

# Start capturing video from the webcam
cap = cv.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define Arabic labels for the classes predicted by the model
labels = {
    0: 'السلام عليكم',
    1: 'كيف الحال',
    2: 'انا طارق'
}

# Initialize variables for tracking signs
predicted_label = "إشارة غير معروفة"
last_detected_sign = None
sign_start_time = None
displayed_text = ""
confirmation_time = 3.0  # Time to confirm a sign
flash_duration = 0.3  # Flash duration in seconds
flash_start_time = None  # To track when to stop the flash

# Load the font for displaying Arabic text
font_path = r"D:\Sign Language Recognition\Fonts\arial.ttf"
font_size = 40  # Adjust the size as needed
font = ImageFont.truetype(font_path, font_size)  # Load the font

# Define styles for hand landmarks and connections
hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)  # Purple circles for landmarks
hand_connection_style = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1)  # Purple lines for connections

# Main loop to process video frames
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture image.")
        break  # Exit if no frame is captured

    frame = cv.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB

    results = hands.process(rgb_frame)  # Process the frame for hand detection

    if results.multi_hand_landmarks:  # Check for detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            data_corr = []  # Clear data_corr for each hand
            x_coords = []
            y_coords = []

            # Draw hand landmarks and connections
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])  # Convert to pixel values
                y = int(landmark.y * frame.shape[0])  # Convert to pixel values
                x_coords.append(x)
                y_coords.append(y)
                data_corr.append(landmark.x)
                data_corr.append(landmark.y)

                # Draw each landmark using the defined style
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                           hand_landmark_style, hand_connection_style)

            # Calculate bounding box for the hand
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Draw bounding box around the hand
            cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Green box

            if len(data_corr) == 42:  # Ensure correct input size
                predicted_class = int(model.predict([np.asarray(data_corr)])[0])  # Predict

                if predicted_class in labels:  # Check if the prediction matches known labels
                    current_sign = labels[predicted_class]  # Retrieve Arabic label

                    # Update sign_start_time only when the current_sign changes
                    if current_sign != last_detected_sign:
                        last_detected_sign = current_sign  # Update last detected sign
                        sign_start_time = time.time()  # Reset the timer for the new sign

                    # Check if the sign has been held long enough
                    if sign_start_time is not None and (time.time() - sign_start_time >= confirmation_time):
                        displayed_text += current_sign + " "  # Confirm and add to displayed text
                        flash_start_time = time.time()  # Start flash effect
                        sign_start_time = None  # Reset the timer for re-detection

            # Trigger the flash effect for the bounding box
            if flash_start_time and (time.time() - flash_start_time <= flash_duration):
                # Flash effect: Fill the bounding box with white color at 75% opacity
                overlay = frame.copy()
                cv.rectangle(overlay, (min_x, min_y), (max_x, max_y), (255, 255, 255), -1)  # White rectangle
                cv.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)  # Flash effect with 75% opacity

            elif flash_start_time and (time.time() - flash_start_time > flash_duration):
                # Stop the flash effect after the specified duration
                flash_start_time = None

    else:
        predicted_label = "إشارة غير معروفة"  # Reset if no hands detected

    # Process displayed text for Arabic rendering
    reshaped_text = arabic_reshaper.reshape(displayed_text)  # Reshape the Arabic text
    bidi_text = get_display(reshaped_text)  # Convert the reshaped text to be displayed from right to left

    # Create a box for the text display with black background and increased height
    box_margin = 10
    text_size = cv.getTextSize(bidi_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    box_x0, box_y0 = 10, 10
    box_x1, box_y1 = box_x0 + text_size[0] + box_margin * 2, box_y0 + text_size[1] * 2 + box_margin * 2  # Increased height

    # Draw the box around the text in black
    cv.rectangle(frame, (box_x0, box_y0), (box_x1, box_y1), (255, 255, 255), -1)  # Black box

    # Draw the reshaped text on the frame in white
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    draw.text((box_x0 + box_margin, box_y0 + box_margin), bidi_text, font = font, fill = (0, 0, 0))  # White text
    frame = np.array(pil_img)  # Convert back to OpenCV format

    # Show the processed frame in a window
    cv.imshow('Sign Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(25) & 0xFF == ord('q'):
        print("User pressed 'q'. Exiting...")
        break

# Release the camera and close all windows
cap.release()
cv.destroyAllWindows()
