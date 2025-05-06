import requests
from PIL import Image
import io

# Your Roboflow workspace, project, and endpoint information
ROBOFLOW_API_KEY = "woQWDmLgxdIEStq4s4ld"
ROBOFLOW_WORKFLOW_API_URL = "https://serverless.roboflow.com/detect-count-and-visualize-2"

def roboflow_predict(img):
    # Convert numpy image to bytes for upload
    img_pil = Image.fromarray(img)
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    response = requests.post(
        f"{ROBOFLOW_WORKFLOW_API_URL}",
        files={"image": buf},
        data={
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    predictions = response.json()
    return predictions

# In your Streamlit code, replace YOLO section with:
st.header("Roboflow Object Detection (Free API)")

if uploaded_file is not None:
    # ...     [process image as above]
    # Place this inside your `if uploaded_file is not None` block
    roboflow_results = roboflow_predict(img)

    if "predictions" in roboflow_results and roboflow_results["predictions"]:
        crack_count = sum(1 for pred in roboflow_results["predictions"] if pred.get("class") == "crack")
        st.write(f"Detected cracks: **{crack_count}**")

        if "output_image" in roboflow_results and roboflow_results["output_image"]:
            # The output image is usually a base64 string in the Roboflow response
            import base64
            output_image = roboflow_results["output_image"]

            st.image(base64.b64decode(output_image), caption="Roboflow Detection", width=700)
    else:
        st.write("No cracks detected.")