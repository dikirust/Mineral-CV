import cv2
import numpy as np
import os
import glob
import re

# Koordinat crop untuk setiap baris (8 baris)
row_coords = [
    (53, 360, 4580, 367),   # row 1
    (53, 740, 4580, 367),   # row 2
    (53, 1120, 4580, 367),  # row 3
    (53, 1500, 4580, 367),  # row 4
    (53, 2550, 4580, 367),  # row 5
    (53, 2930, 4580, 367),  # row 6
    (53, 3310, 4580, 367),  # row 7
    (53, 3690, 4580, 367),  # row 8
]

# Path ke folder datasets
dataset_dir = r"C:\Users\diki.rustian\Documents\GitHub\Mineral-CV\corepic\datasets"
# Ambil hanya file yang berekstensi .jpg (case-insensitive) dan mengandung 'BOX' pada namanya
jpg_files = [
    f for f in glob.glob(os.path.join(dataset_dir, "*"))
    if f.lower().endswith(".jpg") and "box" in os.path.basename(f).lower()
]

# Fungsi untuk ekstrak nomor BOX dari nama file
def extract_box_number(filename):
    match = re.search(r'BOX\s*(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# Urutkan file berdasarkan nomor BOX
jpg_files.sort(key=lambda x: extract_box_number(os.path.basename(x)))

# Simpan linearized dari setiap BOX
all_linearized = []

for img_path in jpg_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"File not found or unreadable: {img_path}")
        continue

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(os.path.dirname(__file__), 'cropped_output', base_name)
    os.makedirs(output_dir, exist_ok=True)

    row_images = []
    for idx, (x, y, w, h) in enumerate(row_coords):
        crop = img[y:y+h, x:x+w]
        out_path = os.path.join(output_dir, f"{base_name}_row_{idx+1}.jpg")
        cv2.imwrite(out_path, crop)
        row_images.append(crop)

    # Linearize (gabungkan horizontal), lalu rotate 90 derajat searah jarum jam
    if row_images:
        linearized = cv2.hconcat(row_images)
        linearized_rotated = cv2.rotate(linearized, cv2.ROTATE_90_CLOCKWISE)
        linearized_path = os.path.join(output_dir, f"{base_name}_linearized.jpg")
        cv2.imwrite(linearized_path, linearized_rotated)
        print(f"Saved: {linearized_path}")
        all_linearized.append(linearized_rotated)

# Setelah semua BOX selesai, gabungkan semua linearized menjadi satu gambar besar (vertikal, bukan horizontal)
if all_linearized:
    print(f"Jumlah gambar linearized yang akan digabung: {len(all_linearized)}")
    for idx, img in enumerate(all_linearized):
        print(f"Linearized ke-{idx+1} shape: {img.shape}")
    # Gabungkan secara vertikal (bottom BOX sebelumnya ke top BOX berikutnya)
    full_linearized = cv2.vconcat(all_linearized)
    # Resize jika terlalu tinggi
    max_height = 10000  # misal, batas tinggi 10.000 piksel
    if full_linearized.shape[0] > max_height:
        scale = max_height / full_linearized.shape[0]
        new_width = int(full_linearized.shape[1] * scale)
        full_linearized = cv2.resize(full_linearized, (new_width, max_height))
    final_output_dir = os.path.join(os.path.dirname(__file__), 'cropped_output')
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, "full_linearized.jpg")
    success = cv2.imwrite(final_output_path, full_linearized)
    if success:
        print(f"Saved: {final_output_path}")
    else:
        print(f"GAGAL menyimpan: {final_output_path}")
else:
    print("Tidak ada gambar linearized yang bisa digabung.")
