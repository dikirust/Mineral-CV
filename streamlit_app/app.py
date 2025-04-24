
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Laterit Mineral Vision", layout="centered")

st.title("🧪 CV Assessment: Karakterisasi Awal Mineral Laterit")
st.markdown("Upload gambar batuan/core, dan sistem akan mencoba mengidentifikasi zona: Limonit, Saprolit, atau Batuan Induk.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

def classify_zone(rgb):
    r, g, b = rgb
    if r > 120 and g < 80:
        return "Zona Limonit (Coklat-Kemerahan)", "✅ Layak"
    elif g > 100 and r < 100:
        return "Zona Saprolit (Hijau Sabun)", "✅ Layak"
    elif r < 80 and g < 80 and b < 80:
        return "Batuan Induk (Gelap/Serpentinit)", "🔍 Perlu Analisa Lanjut"
    else:
        return "Tidak Teridentifikasi", "❌ Tidak Layak"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    img_array = np.array(image.resize((100, 100)))
    mean_color = img_array.mean(axis=(0, 1))[:3]  # Rata-rata RGB
    label, assessment = classify_zone(mean_color)

    st.markdown(f"**Prediksi Zona:** {label}")
    st.markdown(f"**Assessment Awal:** {assessment}")
