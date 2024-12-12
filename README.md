---
title: BCCDo
emoji: 🚀
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

BCC Donation - YOLO Object Detection

This project utilizes a YOLOv10 model for object detection, specifically to detect cells in the BCC Donation dataset. The goal is to preprocess the dataset, fine-tune the YOLO model, perform image augmentation, and deploy the solution via a web app using Streamlit. It also calculates the precision and recall of the model on various classes.

Project Overview
This repository includes the following key features:

Data Preprocessing: Enhancement of purple regions in images to highlight relevant objects (cells).
Data Augmentation: Rotation and cropping of images to improve model performance.
Model Training: Fine-tuning of a YOLOv10 model for object detection.
Evaluation: Calculation of Precision and Recall metrics for object detection accuracy.
Deployment: Web app deployment using Streamlit.

Requirements
Python 3.8 or later
torch (for model training and inference)
ultralytics (for YOLO model)
opencv-python (for image processing)
pandas (for metrics calculation)
scikit-learn (for precision and recall calculation)
streamlit (for web application)
Pillow (for image handling)

Install Dependencies
To install the required dependencies, you can create a virtual environment and run the following command:
pip install -r requirements.txt

Where the requirements.txt file should include:
torch==1.12.1
ultralytics==8.0.54
opencv-python==4.6.0.66
pandas==1.5.2
scikit-learn==1.1.2
streamlit==1.12.0
Pillow==9.0.1

How to Run the Application
Clone the repository: You can clone this repository to your local machine or directly upload it to Hugging Face Spaces.

Run the Streamlit app: After installing the dependencies, run the following command to launch the Streamlit web app:

streamlit run app.py
Upload Image: Once the Streamlit app is running, you can upload an image, and the app will:

Preprocess the image to enhance the purple regions.
Run the YOLOv10 object detection model.
Display the results with bounding boxes and the predicted class.
Calculate and display precision and recall metrics for the detection.

Fine-Tuning the Model
To fine-tune the YOLO model on the BCCD dataset, you need to execute the following code in the train.py script:
from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("yolov10s.pt")

# Fine-tune on the BCCD dataset
model.train(data="bccd.yaml", epochs=10, imgsz=640, batch=16, workers=4)
Ensure that the bccd.yaml file is set up properly to point to your training and validation datasets. This script will train the model for 10 epochs and save the fine-tuned model.

Metrics Calculation
Precision and recall are calculated by comparing the model's predicted results with the ground truth labels.

Deployment
Deploying on Hugging Face Spaces
Create a Hugging Face Space:

Sign up or log in to Hugging Face.
Create a new Space and select Streamlit as the framework.

Upload files:
Go to the Files tab of your Space.
Upload the entire project, including the model files (yolov10s.pt), your Streamlit app (app.py), and other necessary files (such as requirements.txt).
Install dependencies:

Hugging Face Spaces will automatically install the dependencies listed in the requirements.txt file.
Deploy and test:

Once everything is uploaded and dependencies are installed, Hugging Face will automatically deploy the application.
You can now access the web app via the Hugging Face URL.

Acknowledgments
The BCCD Dataset for cell images and annotations.
YOLO for object detection and ultralytics for the YOLOv10 implementation.
BCCD Dataset at https://github.com/Shenggan/BCCD_Dataset
Fine YOLOv10 from https://docs.ultralytics.com/models/yolov10/
