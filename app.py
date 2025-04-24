from ultralytics import YOLO
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tempfile
import cv2
import os
from pytube import YouTube

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO(r"best.pt")

model = load_model()

st.title("🚗 Number Plate Detection")

st.markdown("### Upload image, video or use a YouTube link")

# UI: Uploads
col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
with col2:
    image_url = st.text_input("🌐 Paste image URL")

uploaded_video = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov", "mkv"])
youtube_url = st.text_input("▶️ Paste YouTube Video URL")

image = None

# Load image from upload or URL
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Could not load image: {e}")

# Detect button
if st.button("🔍 Detect Number Plate"):
    # Image detection
    if image:
        with st.spinner("Detecting in image..."):
            results = model(image)
            result_img = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Input Image", use_column_width=True)
        with col2:
            st.image(result_img, caption="✅ Detection Result", use_column_width=True)

    # Local video detection
    elif uploaded_video is not None:
        with st.spinner("Detecting in uploaded video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            tfile.close()
            video_path = tfile.name

    # YouTube video detection
    elif youtube_url:
        with st.spinner("Downloading and processing YouTube video..."):
            try:
                yt = YouTube(youtube_url)
                stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
                video_path = stream.download(output_path=tempfile.gettempdir())
            except Exception as e:
                st.error(f"Failed to download YouTube video: {e}")
                video_path = None

    # Run detection on video (from upload or YouTube)
    if 'video_path' in locals() and video_path:
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_column_width=True)

        cap.release()
        if uploaded_video: os.remove(video_path)  # Only delete if it's a temp upload
