import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets

# Load the Keras model
model = load_model('./model.h5')  # Update with your model path

# Define your classes
classes = ['abnormal', 'normal']

# Function to preprocess the image for EfficientNetB0
def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)  # If needed
    return img

# Function to perform classification and overlay results on the frame
def classify_frame(frame):
    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make prediction
    predictions = model.predict(processed_frame)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions[0])

    # Get the class label
    predicted_class = classes[predicted_class_index]

    # Get the confidence score
    confidence = predictions[0][predicted_class_index]

    # Overlay predictions on the frame
    label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# Path to the video file (update with your file path)
video_path = './demo1.mp4'  # Update with your video path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Define a flag for breaking the loop
stop_button = widgets.Button(description='Stop')
display(stop_button)

def stop_callback(button):
    cap.release()

stop_button.on_click(stop_callback)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Classify the frame
    classified_frame = classify_frame(frame)

    # Convert the frame from BGR to RGB
    classified_frame_rgb = cv2.cvtColor(classified_frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Matplotlib
    plt.imshow(classified_frame_rgb)
    plt.axis('off')
    plt.show()
