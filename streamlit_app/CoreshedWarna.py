import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, measure, color
from scipy import ndimage as ndi
from ultralytics import YOLO
import os
import easyocr
import re
from PIL import Image

# Function to calculate the percentage area of the segmented region
def calculate_segmented_area_percentage(mask):
    """
    Calculate the percentage area of the segmented region.

    Parameters:
    mask (numpy.ndarray): Binary mask where segmented regions are non-zero.

    Returns:
    float: Percentage of the segmented area.
    """
    total_pixels = mask.size  # Total number of pixels in the mask
    segmented_pixels = np.count_nonzero(mask)  # Count of non-zero pixels in the mask
    percentage = (segmented_pixels / total_pixels) * 100  # Calculate percentage
    return percentage

# Get the absolute path of the current script directory
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'best.pt')

# Verify if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

# Debug: Print the resolved model path
print(f"Resolved model path: {model_path}")

# Load YOLOv11 model
model = YOLO(model_path)  # Replace 'yolo11n.pt' with your trained YOLOv11 model

# Streamlit app
st.title("🛠️ Advanced Geoscience Image Analysis for Nickel Mining")

# Add a description with an icon
st.markdown("""
🌍 This application leverages advanced image segmentation and object detection techniques to analyze geoscience images for nickel mining. 
It provides insights into mineral composition, segmented area percentages, and object detection results.
""")

# Consolidate image upload to a single uploader for all analyses
uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image with PIL to handle orientation
    pil_img = Image.open(uploaded_file)
    pil_img = pil_img.convert("RGB")  # Ensure 3 channels
    img = np.array(pil_img)

    # Display the original image
    st.image(img, caption="Uploaded Image", width=700)  # Adjust width as needed
    st.write("\n")  # Add spacing

    # Convert RGBA to RGB if necessary
    if img.shape[-1] == 4:  # Check if the image has 4 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Proceed with segmentation analysis
    st.sidebar.header("Color Segmentation")

    # Define HSV ranges for each color (adjust as needed)
    hsv_presets = {
        "Coklat Kemerahan": ((10, 50, 20), (50, 100, 225)),      # reddish brown
        "Merah Tua": ((0, 80, 20), (10, 100, 225)),            # dark red
        "Oranye": ((20, 80, 20), (35, 100, 225)),             # orange
        "Kuning Kecoklatan": ((35, 60, 20), (50, 100, 225)),   # yellowish brown
        "Hijau Terang": ((90, 40, 20), (140, 90, 225)),       # bright green
        "Hijau Zaitun": ((70, 30, 20), (95, 70, 225)),         # olive green
        "Abu-abu Kehijauan": ((80, 10, 20), (140, 40, 225)),     # greenish gray
        "Coklat Kehijauan": ((45, 40, 20), (85, 70, 225)),     # greenish brown
    }

    color_explanations = {
        "Coklat Kemerahan": "Reddish brown, often indicates laterite or oxidized minerals.",
        "Merah Tua": "Dark red, may represent iron-rich minerals.",
        "Oranye": "Orange, could be limonite or weathered surfaces.",
        "Kuning Kecoklatan": "Yellowish brown, typical of goethite or weathered zones.",
        "Hijau Terang": "Bright green, possibly serpentine or chlorite.",
        "Hijau Zaitun": "Olive green, often olivine or altered ultramafics.",
        "Abu-abu Kehijauan": "Greenish gray, may indicate mixed silicates.",
        "Coklat Kehijauan": "Greenish brown, transitional weathering products.",
    }

    if st.sidebar.button("Proceed Color Segmentation"):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_percentages = {}

        for color_name, (lower, upper) in hsv_presets.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pct = calculate_segmented_area_percentage(mask)
            color_percentages[color_name] = (pct, mask)

        # Show all overlays first
        for color_name, (pct, mask) in color_percentages.items():
            label_overlay = color.label2rgb(mask, image=img, bg_label=0, alpha=0.4)
            st.image(label_overlay, caption=f"{color_name} Overlay", width=350, clamp=True)

        # Show summary explanation after all overlays
        st.markdown("### Color Segmentation Summary")
        for color_name, (pct, mask) in color_percentages.items():
            st.write(f"**% {color_name}:** {pct:.2f}%")
            st.write(f"{color_explanations[color_name]}")
            st.write("\n")
