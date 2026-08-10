import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Bahraini Currency Classifier",
    page_icon="🇧🇭",
    layout="centered"
)

st.title("🇧🇭 Bahraini Currency Recognition System")
st.write("Upload an image or capture a photo using your camera to classify Bahraini banknotes.")

# Load Model and Class Names (Cached to avoid reloading on user interactions)
@st.cache_resource
def load_currency_model():
    # Update path if hosting locally vs Colab/Drive environment
    model_path = "bahrain_currency_model.keras"
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def load_class_names():
    class_names_path = "class_names.json"
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    return class_names

try:
    model = load_currency_model()
    class_names = load_class_names()
    st.sidebar.success("Model & Classes Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Ensure 'bahrain_currency_model.keras' and 'class_names.json' exist in the app directory.")

# File Input Selection Mode
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Select Image Source:", ("Upload File", "Take Photo (Camera)"))

uploaded_file = None

if input_mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Choose a currency image...", 
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("Take a photo of the note")

# Prediction Pipeline
if uploaded_file is not None:
    # Display Input Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)
    
    with st.spinner("Classifying image..."):
        start_time = time.time()
        
        # Preprocessing (Match training: Resize to 224x224 & Normalize /255.0)
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
        
        # Inference
        predictions = model.predict(img_batch, verbose=0)
        inference_time = (time.time() - start_time) * 1000 # convert to ms
        
        # Result Extraction
        predicted_idx = np.argmax(predictions[0])
        predicted_label = class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100

    # Display Top Result
    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric(label="Predicted Currency Class", value=f"{predicted_label}")
    st.progress(confidence / 100.0)
    st.write(f"**Confidence Score:** {confidence:.2f}%")
    st.write(f"**Inference Speed:** {inference_time:.1f} ms")

    # Detailed Class Probabilities Sidebar / Breakdown
    with st.expander("View Full Probability Breakdown"):
        prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        for name, prob in sorted_probs:
            st.write(f"**{name}:** {prob * 100:.2f}%")

# Information / Guidance Section
st.markdown("---")
st.markdown("""
### 💡 Testing & Real-World Guidelines:
* **Background Impact:** Ensure clear contrast between the banknote and the background surface[cite: 9].
* **Distance & Framing:** Centered, clear views yield higher classification confidence[cite: 9].
* **Lighting:** Bright, non-reflective lighting prevents misclassification[cite: 9].
""")