import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import numpy as np
import tempfile

# Load YOLOv5/YOLOv8 model (use "yolov5s.pt" for YOLOv5 or "yolov8n.pt" for YOLOv8)
model = YOLO("yolov8n.pt")  # chto your YOLOv5 model file if needed

st.title("Real-Time Object Detection with YOLO and Streamlit")
st.write("Detect objects live using your camera or upload files.")

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img)
    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

st.header("Live Camera Feed")
webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.header("Upload an Image or Video")
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])
if uploaded_file:
    if uploaded_file.type.startswith('image'):
        import numpy as np
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model(img)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR")
    elif uploaded_file.type.startswith('video'):
        import tempfile, cv2
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")
        cap.release()             