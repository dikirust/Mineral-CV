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

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    img = io.imread(uploaded_file)
    
    # Display the original image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Convert RGBA to RGB if necessary
    if img.shape[-1] == 4:  # Check if the image has 4 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert to HSV and create a mask
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 20, 20), (50, 255, 255))
    
    # Perform binary closing
    closed_mask = ndi.binary_closing(mask, structure=np.ones((7, 7)))
    
    # Label the segmented regions
    label_image = measure.label(closed_mask)
    
    # Generate the label overlay
    image_label_overlay = color.label2rgb(label_image, image=img)
    
    # Display the segmented image
    st.image(image_label_overlay, caption="Segmented Image", use_column_width=True)
    
    # Calculate the segmented area percentage
    segmented_area_percentage = calculate_segmented_area_percentage(closed_mask)
    st.write(f"Segmented Area Percentage: {segmented_area_percentage:.2f}%")