```python
import json
import time

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Bahraini Currency Classifier",
    page_icon="🇧🇭",
    layout="centered"
)


# ============================================================
# App Title
# ============================================================

st.title("🇧🇭 Bahraini Currency Recognition System")

st.write(
    "Upload an image or capture a photo using your camera "
    "to classify Bahraini banknotes and coins."
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_currency_model():
    model_path = "bahrain_currency_model.keras"

    model = tf.keras.models.load_model(model_path)

    return model


# ============================================================
# Load Class Names
# ============================================================

@st.cache_data
def load_class_names():
    class_names_path = "class_names.json"

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Handle either:
    # ["BD 0.500", "BD 1", ...]
    # or
    # {"0": "BD 0.500", "1": "BD 1", ...}

    if isinstance(class_names, dict):
        # Sort dictionary by numeric index if possible
        try:
            class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]
        except (KeyError, ValueError):
            class_names = list(class_names.values())

    return class_names


# ============================================================
# Load Model and Classes
# ============================================================

model = None
class_names = None

try:
    model = load_currency_model()
    class_names = load_class_names()

    st.sidebar.success("✅ Model & Classes Loaded Successfully!")

except Exception as e:

    st.sidebar.error("❌ Failed to load model")

    st.error(f"Error loading model files: {e}")

    st.info(
        "Ensure that 'bahrain_currency_model.keras' and "
        "'class_names.json' exist in the same directory as your Streamlit app."
    )


# ============================================================
# Input Options
# ============================================================

st.sidebar.header("📷 Input Options")

input_mode = st.sidebar.radio(
    "Select Image Source:",
    ("Upload File", "Take Photo (Camera)")
)

uploaded_file = None


# ============================================================
# Handle Image Input
# ============================================================

if input_mode == "Upload File":

    uploaded_file = st.file_uploader(
        "Choose a currency image...",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded_file = st.camera_input(
        "Take a photo of the currency note or coin"
    )


# ============================================================
# Prediction Pipeline
# ============================================================

if uploaded_file is not None:

    # Make sure model was loaded successfully
    if model is None or class_names is None:

        st.error(
            "The model could not be loaded. "
            "Please check the model and class names files."
        )

    else:

        try:

            # ------------------------------------------------
            # Open and Display Image
            # ------------------------------------------------

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Input Image",
                width="stretch"
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with st.spinner("Classifying image..."):

                start_time = time.time()

                # Resize image to 224 x 224
                img_resized = ImageOps.fit(
                    image,
                    (224, 224),
                    Image.Resampling.LANCZOS
                )

                # Convert image to NumPy array
                img_array = np.asarray(
                    img_resized,
                    dtype=np.float32
                )

                # Normalize pixel values to [0, 1]
                img_array = img_array / 255.0

                # Add batch dimension
                # Shape: (224, 224, 3)
                #      -> (1, 224, 224, 3)

                img_batch = np.expand_dims(
                    img_array,
                    axis=0
                )

                # ------------------------------------------------
                # Model Inference
                # ------------------------------------------------

                predictions = model.predict(
                    img_batch,
                    verbose=0
                )

                inference_time = (
                    time.time() - start_time
                ) * 1000


                # ------------------------------------------------
                # Extract Prediction
                # ------------------------------------------------

                probabilities = predictions[0]

                predicted_idx = int(
                    np.argmax(probabilities)
                )

                # Check that class index exists
                if predicted_idx >= len(class_names):

                    st.error(
                        "The number of model output classes does not "
                        "match the number of class names."
                    )

                    st.stop()

                predicted_label = class_names[predicted_idx]

                confidence = (
                    float(probabilities[predicted_idx])
                    * 100
                )


            # ====================================================
            # Display Prediction Results
            # ====================================================

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            st.metric(
                label="Predicted Currency Class",
                value=str(predicted_label)
            )

            st.progress(
                min(max(confidence / 100.0, 0.0), 1.0)
            )

            st.write(
                f"**Confidence Score:** {confidence:.2f}%"
            )

            st.write(
                f"**Inference Speed:** {inference_time:.1f} ms"
            )


            # ====================================================
            # Full Probability Breakdown
            # ====================================================

            with st.expander(
                "📊 View Full Probability Breakdown"
            ):

                prob_dict = {
                    class_names[i]: float(probabilities[i])
                    for i in range(
                        min(
                            len(class_names),
                            len(probabilities)
                        )
                    )
                }

                sorted_probs = sorted(
                    prob_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for name, prob in sorted_probs:

                    st.write(
                        f"**{name}:** {prob * 100:.2f}%"
                    )

                    st.progress(
                        min(max(float(prob), 0.0), 1.0)
                    )


        except Exception as e:

            st.error(
                f"An error occurred while processing the image: {e}"
            )


# ============================================================
# Information & Usage Guidelines
# ============================================================

st.markdown("---")

st.markdown(
    """
### 💡 Guidelines for Best Results

- **Background Contrast:** Place the banknote or coin on a plain,
  uncluttered surface for cleaner feature extraction.
- **Lighting:** Ensure the currency is well lit and avoid harsh
  reflections or glare.
- **Framing:** Center the currency item and make sure it is fully
  visible in the image.
- **Image Quality:** Use a clear, sharp image for better classification.
- **Camera:** When using the camera option, hold the phone steady
  and ensure the currency occupies a reasonable portion of the frame.
"""
)
```
