import cv2
import numpy as np
import os  # Import os module
import glob

# Load image
img = cv2.imread(r"C:\Users\diki.rustian\Documents\GitHub\Mineral-CV\corepic\N161927_10.00M-14.00M_BOX 5-6.JPG")
if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Define your 8 specific (x, y, w, h) for each row
row_coords = [
    (53, 360, 4580, 367),   # row 1
    (53, 740, 4580, 367),   # row 2
    (53, 1120, 4580, 367),   # row 3
    (53, 1500, 4580, 367),   # row 4
    (53, 2550, 4580, 367),  # row 5
    (53, 2930, 4580, 367),  # row 6
    (53, 3310, 4580, 367),  # row 7
    (53, 3690, 4580, 367),  # row 8
]

# Get base name without extension
base_name = os.path.splitext(os.path.basename(r"C:\Users\diki.rustian\Documents\GitHub\Mineral-CV\corepic\N161927_10.00M-14.00M_BOX 5-6.JPG"))[0]
# Create subfolder inside cropped_output with base_name
output_dir = os.path.join(os.path.dirname(__file__), 'cropped_output', base_name)
os.makedirs(output_dir, exist_ok=True)

for idx, (x, y, w, h) in enumerate(row_coords):
    crop = img[y:y+h, x:x+w]
    out_path = os.path.join(output_dir, f"{base_name}_row_{idx+1}.jpg")
    cv2.imwrite(out_path, crop)

# Linearize cropped rows into one image and rotate 90 degrees clockwise

# Get all cropped row images in order
row_images = []
for i in range(1, 9):
    row_path = os.path.join(output_dir, f"{base_name}_row_{i}.jpg")
    row_img = cv2.imread(row_path)
    if row_img is not None:
        row_images.append(row_img)

if row_images:
    # Concatenate images horizontally (right edge to left edge)
    linearized = cv2.hconcat(row_images)
    # Rotate 90 degrees clockwise
    linearized_rotated = cv2.rotate(linearized, cv2.ROTATE_90_CLOCKWISE)
    # Save the result
    linearized_path = os.path.join(output_dir, f"{base_name}_linearized.jpg")
    cv2.imwrite(linearized_path, linearized_rotated)
