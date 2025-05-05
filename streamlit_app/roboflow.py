import streamlit as st
import requests
from PIL import Image
import io
import base64
import numpy as np

# Your Roboflow workspace, project, and endpoint information
ROBOFLOW_API_KEY = "woQWDmLgxdIEStq4s4ld"
ROBOFLOW_WORKFLOW_API_URL = "https://serverless.roboflow.com/detect-count-and-visualize-2"

# Add a slider for confidence threshold
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

def roboflow_predict(img, confidence):
    # Convert numpy image to bytes for upload
    img_pil = Image.fromarray(img)
    if img_pil.mode == "RGBA":
        img_pil = img_pil.convert("RGB")  # Convert RGBA to RGB
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    response = requests.post(
        f"{ROBOFLOW_WORKFLOW_API_URL}",
        files={"image": buf},
        data={
            "api_key": ROBOFLOW_API_KEY,
            "confidence": confidence,  # Include confidence threshold
        },
    )

    # Check if the response is successful
    if response.status_code == 200:
        try:
            predictions = response.json()
        except ValueError:
            predictions = {"error": "Invalid JSON response from the API."}
    else:
        predictions = {"error": f"API request failed with status code {response.status_code}: {response.text}"}

    return predictions

# Streamlit app
st.title("Roboflow Object Detection App")
st.write("Upload an image to detect objects using the Roboflow API.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # Call the Roboflow API with confidence threshold
    with st.spinner("Processing image..."):
        roboflow_results = roboflow_predict(img_array, confidence_threshold)

    # Display results
    if "predictions" in roboflow_results and roboflow_results["predictions"]:
        crack_count = sum(1 for pred in roboflow_results["predictions"] if pred.get("class") == "crack")
        st.write(f"Detected cracks: **{crack_count}**")

        if "output_image" in roboflow_results and roboflow_results["output_image"]:
            output_image = roboflow_results["output_image"]
            st.image(base64.b64decode(output_image), caption="Roboflow Detection", use_column_width=True)
    else:
        st.write("No objects detected.")

    # Display raw API response for debugging
    st.subheader("Raw API Response")
    st.json(roboflow_results)