import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import warnings

# Suppress warnings related to deprecated parameters
warnings.filterwarnings("ignore", category=UserWarning, message="The `use_column_width` parameter has been deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Please use `torch.amp.autocast('cuda', args...)` instead")

# Load YOLO model (Replace with your trained model path if needed)
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" or "yolov8m.pt" for better accuracy

# Streamlit UI
st.title("🏥 Object Detectio  Using Yolov8")
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Upload video or use webcam
source_option = st.sidebar.radio("Select Source:", ("Webcam", "Upload Video"))

if source_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video_path)
    else:
        cap = None
else:
    cap = cv2.VideoCapture(0)  # Webcam

# Streamlit Video Display
stframe = st.empty()

# Process Video Stream
if cap is not None:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video has ended or no frame available.")
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform Object Detection
        results = model(frame_rgb)[0]

        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = results.names[int(box.cls[0])]
            if conf > confidence_threshold:
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display processed frame
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

st.sidebar.success("✅ System is Running!")
