import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")

# 🌿 Custom CSS Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Lexend', sans-serif;
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            color: #f0f8ff;
        }

        .main-header {
            text-align: center;
            padding: 2rem 1rem;
        }

        .main-header h1 {
            color: #7CFC00;
            font-size: 3rem;
        }

        .main-header p {
            color: #dcdcdc;
            font-size: 1.1rem;
        }

        .card {
            background-color: rgba(255,255,255,0.05);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            margin: 2rem auto;
            max-width: 800px;
        }

        .stFileUploader {
            border: 2px dashed #7CFC00 !important;
            background-color: #1e1e2f !important;
            border-radius: 10px;
            padding: 1.5rem;
        }

        .stButton > button {
            background-color: #7CFC00;
            color: black;
            font-weight: 600;
            border: none;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# 🌿 Load model
try:
    model = load_model(r"project/model.h5")  # ✅ adjust path to model
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# 🌿 Class labels and treatment
class_labels = {
    0: 'Bacterial_spot',
    1: 'Early_blight',
    2: 'Healthy',
    3: 'Late_blight',
    4: 'Leaf_rust',
    5: 'Mosaic_virus',
    6: 'Leaf_curl',
    7: 'Powdery_mildew'
}

treatment_info = {
    'Bacterial_spot': ("Dark, water-soaked lesions on leaves.", "Apply copper fungicide for 1–2 weeks."),
    'Early_blight': ("Brown concentric rings on older leaves.", "Use mancozeb or chlorothalonil."),
    'Healthy': ("This plant appears to be healthy.", "Maintain proper watering and sunlight."),
    'Late_blight': ("Dark brown patches with white mold.", "Remove infected plants, use fungicides."),
    'Leaf_rust': ("Rusty orange pustules on underside.", "Use sulfur spray 2x a week."),
    'Mosaic_virus': ("Mottled leaf color, stunted growth.", "No cure. Remove infected plant."),
    'Leaf_curl': ("Leaves curl upward or downward.", "Spray neem oil weekly."),
    'Powdery_mildew': ("White powdery coating on leaves.", "Use potassium bicarbonate spray.")
}

# 🌿 Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("<h1>🌿 Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a leaf image to detect plant diseases and get treatment information.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 📤 Upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Leaf Image (jpg / png)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# 🔍 Prediction
if uploaded_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    image_display = Image.open(uploaded_file)

    # 📸 Show resized uploaded image (centered)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image_display, caption="🖼 Uploaded Leaf", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # 🧪 Preprocessing
    img = image_display.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 🔮 Prediction
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]
    confidence = round(float(np.max(prediction) * 100), 2)

    # 📊 Display Results
    st.markdown(f"### 🧬 Prediction: `{predicted_label}`")
    st.progress(confidence / 100)
    st.markdown(f"**Confidence Score:** `{confidence}%`")
    st.markdown("---")
    st.markdown(f"**🩺 Symptom:** {treatment_info[predicted_label][0]}")
    st.markdown(f"**💊 Treatment:** {treatment_info[predicted_label][1]}")

    st.markdown('</div>', unsafe_allow_html=True)
