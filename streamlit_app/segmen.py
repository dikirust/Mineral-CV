import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, measure, color
from scipy import ndimage as ndi

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

# Streamlit app
st.title("Image Segmentation and Area Calculation")

# Add sliders for HSV range adjustment
st.sidebar.header("Mask HSV Range Adjustment")
h_min = st.sidebar.slider("Hue Min", 0, 179, 20)
h_max = st.sidebar.slider("Hue Max", 0, 179, 50)
s_min = st.sidebar.slider("Saturation Min", 0, 255, 20)
s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
v_min = st.sidebar.slider("Value Min", 0, 255, 20)
v_max = st.sidebar.slider("Value Max", 0, 255, 255)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    img = io.imread(uploaded_file)
    
    # Display the original image
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Convert RGBA to RGB if necessary
    if img.shape[-1] == 4:  # Check if the image has 4 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Add a button to confirm HSV value adjustments
    if st.sidebar.button("Proceed with HSV Values"):
        # Convert to HSV and create a mask
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # Display the raw mask (before binary closing)
        st.image(mask, caption="Raw Mask (Before Binary Closing)", use_container_width=True, clamp=True)

        # Use the raw mask directly
        closed_mask_uint8 = mask

        # Create the remaining unmasked image
        unmasked_image = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(closed_mask_uint8))

        # Label the segmented regions
        label_image = closed_mask_uint8

        # Generate the label overlay
        image_label_overlay = color.label2rgb(label_image, image=img)

        # Display the segmented image
        st.image(image_label_overlay, caption="Segmented Image", use_container_width=True)

        # Calculate the segmented area percentage
        segmented_area_percentage = calculate_segmented_area_percentage(closed_mask_uint8)
        remaining_unsegmented_percentage = 100 - segmented_area_percentage
        st.write(f"Segmented Area Percentage: {segmented_area_percentage:.2f}%")
        st.write(f"Remaining Unsegmented Area Percentage: {remaining_unsegmented_percentage:.2f}%")

        # Display the remaining unmasked image
        st.image(unmasked_image, caption="Remaining Unmasked Image", use_container_width=True)