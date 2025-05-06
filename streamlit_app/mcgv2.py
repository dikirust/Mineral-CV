import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, measure, color
from scipy import ndimage as ndi
from ultralytics import YOLO
import os

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
    st.sidebar.header("Mineral Selection")

    # Define HSV ranges for each material
    hsv_ranges = {
        "Nikel": (20, 50, 20, 255, 20, 255),
        "Besi": (0, 30, 0, 50, 50, 100),  # Dummy values for grayish range
        "Sulfur": (10, 40, 10, 60, 60, 120)  # Dummy values for grayish range
    }

    # Update material selection options to include 'Specific Color'
    selected_material = st.sidebar.radio(
        "Select Mineral:",
        options=["Nikel", "Besi", "Sulfur", "Specific Color"],
        index=0
    )

    # Define default HSV values for other materials
    if selected_material != "Specific Color":
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_ranges[selected_material]

    # Add sliders for HSV range adjustment, active only for 'Specific Color'
    if selected_material == "Specific Color":
        st.sidebar.header("Mask HSV Range Adjustment")
        h_min = st.sidebar.slider("Hue Min", 0, 179, 20)
        h_max = st.sidebar.slider("Hue Max", 0, 179, 50)
        s_min = st.sidebar.slider("Saturation Min", 0, 255, 20)
        s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
        v_min = st.sidebar.slider("Value Min", 0, 255, 20)
        v_max = st.sidebar.slider("Value Max", 0, 255, 255)

    # Process the image based on the selected material
    if selected_material:
        # Add a button to confirm HSV value adjustments
        if st.sidebar.button("Proceed Mineral Prediction"):
            # Convert to HSV and create a mask
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            st.image(mask, caption=f"Mask for {selected_material}", width=700, clamp=True)
            st.write("\n")  # Add spacing

            # Use the raw mask directly
            closed_mask_uint8 = mask

            # Create the remaining unmasked image
            unmasked_image = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(closed_mask_uint8))

            # Label the segmented regions
            label_image = closed_mask_uint8

            # Generate the label overlay
            image_label_overlay = color.label2rgb(label_image, image=img)

            # Display the segmented image
            st.image(image_label_overlay, caption=f"Segmented {selected_material}", width=700)  # Adjust width as needed
            st.write("\n")  # Add spacing

            # Calculate the segmented area percentage
            segmented_area_percentage = calculate_segmented_area_percentage(closed_mask_uint8)
            remaining_unsegmented_percentage = 100 - segmented_area_percentage
            st.write(f"Segmented area for {selected_material}: **{segmented_area_percentage:.2f}%**")
            st.write("\n")  # Add spacing

            st.write(f"Unsegmented area: **{remaining_unsegmented_percentage:.2f}%**")
            st.write("\n")  # Add spacing

            # Display the remaining unmasked image
            st.image(unmasked_image, caption=f"Unmasked Image", width=700)
            st.write("\n")  # Add spacing

    # Proceed with YOLOv11 object detection
    st.header("YOLOv11 Object Detection")

    # Add a slider for confidence level
    conf_level = st.slider("Confidence Level", min_value=0.0, max_value=1.0, value=0.02, step=0.01)

    results = model.predict(source=img, save=False, conf=conf_level)  # Use slider value for confidence threshold

    for result in results:
        annotated_img = result.plot()  # Annotate image with bounding boxes
        st.image(annotated_img, caption="YOLOv11 Detection", width=700)  # Adjust width as needed
        st.write("\n")  # Add spacing

    # Fixing the AttributeError by accessing model.names as a dictionary
    mechanical_crack_class_id = [key for key, value in model.names.items() if value == 'mechanical crack']
    natural_crack_class_id = [key for key, value in model.names.items() if value == 'natural crack']

    # Debug: Print class IDs for mechanical and natural cracks
    # st.write("Mechanical crack class ID:", mechanical_crack_class_id)
    # st.write("Natural crack class ID:", natural_crack_class_id)

    mechanical_crack_count = 0
    natural_crack_count = 0

    if mechanical_crack_class_id:
        mechanical_crack_class_id = mechanical_crack_class_id[0]
        mechanical_crack_count = sum(1 for box in results[0].boxes if box.cls == mechanical_crack_class_id)

    if natural_crack_class_id:
        natural_crack_class_id = natural_crack_class_id[0]
        natural_crack_count = sum(1 for box in results[0].boxes if box.cls == natural_crack_class_id)

    # Display the counts
    st.write(f"Detected mechanical cracks: **{mechanical_crack_count}**")
    st.write(f"Detected natural cracks: **{natural_crack_count}**")