# Real-Time-object-Detection-App
# What i built ?
I built this project to make object detection accessible in a simple web interface, without requiring advanced setup. By combining YOLO with Streamlit, anyone can interactively try live object detection directly from their browser.

# Why i built it ?
To learn and implement real-time deep learning inference using YOLO models.

To integrate computer vision models into a user-friendly web-based interface.

To explore deployment-ready AI solutions that can be used in various applications like surveillance, retail analytics, traffic monitoring, etc.

To provide a simple framework anyone can reuse and extend.

# How i built it ?
YOLO (Ultralytics) for object detection

Used the pretrained yolov8n.pt model for fast and lightweight inference.

The model outputs bounding boxes and labels for each detected object.

Streamlit for UI

Designed an interface with two sections:

Live Camera Feed (with WebRTC support).

File Uploader for testing images or videos.

WebRTC for real-time video

Integrated streamlit-webrtc for capturing live camera input.

Applied the YOLO detection frame-by-frame for real-time results.

Computer Vision (OpenCV + NumPy)

Handled image/video decoding, frame manipulation, and visualization.
Streamlit Components

st.image and stframe display annotated detection results.

st.file_uploader allows user to upload and test files.

# Features :
Run YOLO object detection directly in your browser.

Live camera detection using WebRTC.

Upload your own images/videos for detection.

Real-time annotated results with bounding boxes and labels.

# Installation :
Clone the repository

git clone https://github.com/Manisha2704860/Real-Time-Object-Detection-App.git
cd realtime-object-detection
Install dependencies

    pip install streamlit streamlit-webrtc ultralytics opencv-python-headless numpy av
Run the app

    streamlit run app.py
Required Libraries and Why
streamlit → Web UI framework for creating interactive apps.

streamlit-webrtc → Enables real-time webcam streaming and frame processing.

ultralytics → Provides YOLOv5/YOLOv8 models for object detection.

opencv-python → Image and video handling.

numpy → Efficient numerical computation for arrays and image processing.

av → Video frame processing needed for WebRTC integration.

# Usage :
Start the app using streamlit run app.py.

Open the link shown in your terminal (http://localhost:8501).

# Choose between:

Live Camera Feed: Click allow camera permission, see live detections.

Upload an Image/Video: Upload custom file for detection.

# Future Improvements :
Add ability to download annotated results (image/video with detected bounding boxes).

Add GPU/CUDA support for faster inference.

Extend support to custom-trained YOLO models.

Deploy online (e.g., Streamlit Cloud, AWS, GCP).
