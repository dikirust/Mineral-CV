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
    # Read the uploaded image
    img = io.imread(uploaded_file)

    # Display the original image
    st.image(img, caption="Uploaded Image", width=700)  # Adjust width as needed
    st.write("\n")  # Add spacing

    # Convert RGBA to RGB if necessary
    if img.shape[-1] == 4:  # Check if the image has 4 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Proceed with segmentation analysis
    st.sidebar.header("Material Selection")

    # Define initial HSV ranges for Tanah (brown) and Batu (dark grey)
    hsv_presets = {
        "Tanah": (0, 33, 150, 255, 20, 255),   # Brownish HSV range (adjust as needed)
        "Batu": (60, 211, 4, 28, 20, 189)        # Dark grey HSV range (adjust as needed)
    }

    selected_material = st.sidebar.radio(
        "Select Material:",
        options=["Tanah", "Batu"],
        index=0
    )

    # Manual HSV sliders for each material
    st.sidebar.header(f"HSV Range for {selected_material}")
    preset = hsv_presets[selected_material]
    h_min = st.sidebar.slider("Hue Min", 0, 179, preset[0])
    h_max = st.sidebar.slider("Hue Max", 0, 179, preset[1])
    s_min = st.sidebar.slider("Saturation Min", 0, 255, preset[2])
    s_max = st.sidebar.slider("Saturation Max", 0, 255, preset[3])
    v_min = st.sidebar.slider("Value Min", 0, 255, preset[4])
    v_max = st.sidebar.slider("Value Max", 0, 255, preset[5])

    # Add a button to confirm HSV value adjustments
    if st.sidebar.button("Proceed Material Segmentation"):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Always create both masks
        tanah_mask = cv2.inRange(
            hsv,
            (hsv_presets["Tanah"][0], hsv_presets["Tanah"][2], hsv_presets["Tanah"][4]),
            (hsv_presets["Tanah"][1], hsv_presets["Tanah"][3], hsv_presets["Tanah"][5])
        )
        batu_mask = cv2.inRange(
            hsv,
            (hsv_presets["Batu"][0], hsv_presets["Batu"][2], hsv_presets["Batu"][4]),
            (hsv_presets["Batu"][1], hsv_presets["Batu"][3], hsv_presets["Batu"][5])
        )

        # Show overlays
        tanah_overlay = color.label2rgb(tanah_mask, image=img, bg_label=0, alpha=0.4)
        batu_overlay = color.label2rgb(batu_mask, image=img, bg_label=0, alpha=0.4)
        st.image(tanah_overlay, caption="Tanah Overlay", width=350, clamp=True)
        st.image(batu_overlay, caption="Batu Overlay", width=350, clamp=True)

        # Calculate area percentages and show masks
        tanah_pct = calculate_segmented_area_percentage(tanah_mask)
        batu_pct = calculate_segmented_area_percentage(batu_mask)
        total = tanah_pct + batu_pct
        if total > 0:
            tanah_pct = (tanah_pct / total) * 100
            batu_pct = (batu_pct / total) * 100
        else:
            tanah_pct = batu_pct = 0

        # st.image(tanah_mask, caption="Tanah Mask", width=350, clamp=True)
        # st.image(batu_mask, caption="Batu Mask", width=350, clamp=True)
        st.write(f"**% Tanah:** {tanah_pct:.2f}%")
        st.write(f"**% Batu:** {batu_pct:.2f}%")
        st.write(f"**Total:** {tanah_pct + batu_pct:.2f}%")