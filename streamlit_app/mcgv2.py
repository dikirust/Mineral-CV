import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, measure, color
from scipy import ndimage as ndi
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

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

# Load YOLOv11 model
model = YOLO('best.pt')  # Replace 'yolo11n.pt' with your trained YOLOv11 model

# Streamlit app
st.title("Image Segmentation and Area Calculation")

# Consolidate image upload to a single uploader for all analyses
uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    img = io.imread(uploaded_file)

    # Display the original image
    st.image(img, caption="Uploaded Image", width=700)  # Adjust width as needed

    # Convert RGBA to RGB if necessary
    if img.shape[-1] == 4:  # Check if the image has 4 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Proceed with segmentation analysis
    st.sidebar.header("Mask HSV Range Adjustment")
    h_min = st.sidebar.slider("Hue Min", 0, 179, 20)
    h_max = st.sidebar.slider("Hue Max", 0, 179, 50)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 20)
    s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
    v_min = st.sidebar.slider("Value Min", 0, 255, 20)
    v_max = st.sidebar.slider("Value Max", 0, 255, 255)

    if st.sidebar.button("Proceed with HSV Values"):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        st.image(mask, caption="Raw Mask (Before Binary Closing)", width=700, clamp=True)

        # Use the raw mask directly
        closed_mask_uint8 = mask

        # Create the remaining unmasked image
        unmasked_image = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(closed_mask_uint8))

        # Label the segmented regions
        label_image = closed_mask_uint8

        # Generate the label overlay
        image_label_overlay = color.label2rgb(label_image, image=img)

        # Display the segmented image
        st.image(image_label_overlay, caption="Segmented Image", width=700)  # Adjust width as needed

        # Calculate the segmented area percentage
        segmented_area_percentage = calculate_segmented_area_percentage(closed_mask_uint8)
        remaining_unsegmented_percentage = 100 - segmented_area_percentage
        st.write(f"Segmented Area Percentage: {segmented_area_percentage:.2f}%")
        st.write(f"Remaining Unsegmented Area Percentage: {remaining_unsegmented_percentage:.2f}%")

        # Display the remaining unmasked image
        st.image(unmasked_image, caption="Remaining Unmasked Image", width=700)

    # Proceed with YOLOv11 object detection
    st.header("YOLOv11 Object Detection")

    # Add a slider for confidence level
    conf_level = st.slider("Confidence Level", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

    # Debug: Print model names to verify class names
    st.write("Model class names:", model.names)

    # Debug: Print input image shape
    st.write("Input image shape:", img.shape)

    results = model.predict(source=img, save=False, conf=conf_level)  # Use slider value for confidence threshold

    # Debug: Print raw results from YOLO model
    st.write("Raw YOLO results:", results)

    for result in results:
        annotated_img = result.plot()  # Annotate image with bounding boxes
        st.image(annotated_img, caption="Detection Results", width=700)  # Adjust width as needed

    # Fixing the AttributeError by accessing model.names as a dictionary
    crack_class_id = [key for key, value in model.names.items() if value == 'crack']

    # Debug: Print crack class ID
    st.write("Crack class ID:", crack_class_id)

    if crack_class_id:
        crack_class_id = crack_class_id[0]
        crack_count = sum(1 for box in results[0].boxes if box.cls == crack_class_id)

        # Debug: Print detected boxes for crack class
        st.write("Detected boxes for crack class:", [box for box in results[0].boxes if box.cls == crack_class_id])
    else:
        crack_count = 0
    st.write(f"Number of cracks detected: {crack_count}")