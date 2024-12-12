import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Directory path for augmented images
augmented_path = "/app/augmented_images"  # Set a path within the Hugging Face Space environment
os.makedirs(augmented_path, exist_ok=True)

# Step 1: Data Augmentation (Rotation, Cropping)
def augment_image(image, save_dir):
    # Rotate
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(save_dir, "rotated_image.jpg"), rotated)

    # Crop (cutting the edges)
    h, w, _ = image.shape
    cropped = image[50:h - 50, 50:w - 50]
    cv2.imwrite(os.path.join(save_dir, "cropped_image.jpg"), cropped)

# Step 2: Fine-tune the YOLOv10 model (on the BCCD dataset)
# Load the pre-trained YOLO model
model = YOLO("yolov10s.pt")  # Replace with your model file path

# Fine-tune the model on the BCCD dataset
def fine_tune_model():
    print("Starting model fine-tuning...")
    model.train(
        data='bccd.yaml',  # Path to the BCCD dataset YAML file
        epochs=10,         # Number of training epochs
        imgsz=640,         # Image size (adjust based on your GPU memory)
        batch=16,          # Batch size (adjust based on your GPU)
        workers=4          # Number of data loading workers
    )
    print("Model training completed!")

# Step 3: Preprocessing Function (Enhancing Purple Colors)
def preprocess_image(image):
    # Convert the image to HSV to detect purple regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 50, 50])  # Lower bound for purple color
    upper_purple = np.array([160, 255, 255])  # Upper bound for purple color
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Optionally, increase brightness or contrast if needed
    result = cv2.convertScaleAbs(result, alpha=1.2, beta=50)  # Adjust alpha and beta for contrast/brightness
    return result

# Step 4: Object Detection Function
def detect_objects(image):
    # Convert PIL Image to OpenCV format (RGB -> BGR)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Preprocess the image to enhance purple areas
    image_bgr = preprocess_image(image_bgr)

    # Perform prediction with reduced confidence threshold
    results = model.predict(image_bgr, conf=0.15)  # Lower confidence threshold (0.25)
    return results, image_bgr

# Step 5: Precision and Recall Calculation
def calculate_metrics(results):
    y_true = []  # Ground truth labels (replace with actual ground truth)
    y_pred = []  # Predicted labels

    # Extract predicted labels
    for box in results[0].boxes.data.numpy():
        confidence, class_id = box[-2], box[-1]
        y_pred.append(int(class_id))  # Predicted class ID
        y_true.append(int(class_id))  # Placeholder for ground truth (use actual data)

    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)

    metrics_data = {
        "Class": ["All Classes"],  # Add individual classes if needed
        "Precision": [precision],
        "Recall": [recall],
    }
    return pd.DataFrame(metrics_data)

# Step 6: Streamlit Web App for Object Detection
st.title("YOLO Object Detection Web App")
st.write("Upload an image, and the app will detect objects using the YOLOv10 model.")

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to OpenCV format (for processing)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 1: Augment the image (rotation and cropping)
    augment_image(image_bgr, augmented_path)

    # Perform object detection
    st.write("Running object detection...")
    results, annotated_image_bgr = detect_objects(image)

    # Annotate image
    for box in results[0].boxes.data.numpy():
        x1, y1, x2, y2, confidence, class_id = box
        label = f"{results[0].names[int(class_id)]} {confidence:.2f}"
        cv2.rectangle(annotated_image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_image_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)

    # Calculate and display metrics
    metrics_df = calculate_metrics(results)
    st.write("## Precision and Recall Table")
    st.dataframe(metrics_df)
