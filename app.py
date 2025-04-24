from ultralytics import YOLO
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO(r"D:\Python\Nepali Vehicles Number Plate Dataset\Number_Plate_Detection-20250423T154434Z-001\Number_Plate_Detection\Napr24\Napr24\runs\detect\train2\weights\best.pt")

model = load_model()

st.title("🚗 Number Plate Detection")

# Image input options
st.markdown("### 📤 Upload Image or Paste Image URL")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with col2:
    image_url = st.text_input("Enter Image URL")

image = None

# Load image from upload or URL
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Could not load image: {e}")

# Detection button
if image:
    if st.button("🔍 Detect Number Plate"):
        with st.spinner("Detecting..."):
            results = model(image)
        result_img = results[0].plot()  # Annotated image

        # Show side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Input Image", use_column_width=True)
        with col2:
            st.image(result_img, caption="✅ Detection Result", use_column_width=True)
