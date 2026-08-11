# 🇧🇭 Bahraini Currency Recognition System

A Deep Learning computer vision system developed to classify Bahraini currency banknotes and coins across various denominations. Built using TensorFlow/Keras and deployed via an interactive Streamlit application supporting image uploads and live webcam captures.

---

## App Features:

- **Dual Input Modes:** Toggle between file upload (.jpg, .png, .jpeg) and live camera capture via device webcam.

- **Human-Readable Display:** Translates raw directory labels into clear banknote and coin titles.

- **Latency & Confidence Tracking:** Real-time feedback showing prediction confidence (%) and inference speed (ms).

- **Probability Breakdown:** Interactive expander displaying full per-class confidence scores.

---

## 📓 Notebook Workflows
**1. Data Generation (GeneratingImages.ipynb)**

Processes raw, unaugmented images from data/pictures_mixed/ through an offline image augmentation pipeline (rotations, brightness adjustments, flips, and scaling) to create a balanced, augmented dataset stored in data/bahrain_currency.zip.

**2. Model Training & Evaluation (Currency_Recognition_DL.ipynb)**

1. **Data Ingestion:** Loads images from data/bahrain_currency.zip with an 80/20 train/validation split.

2. **Preprocessing:** Rescaling pixel values to [0, 1].

3. **Class Weighting:** Uses sklearn.utils.compute_class_weight to address class balance.

4. **Model Architecture:** Trains and compares Sequential and Functional Deep Neural Networks.

5. **Evaluation:** Computes loss/accuracy curves, classification reports (Precision, Recall, F1), confusion matrix heatmaps, and visual error analysis grids.

6. **Model Artifact Export:** Exports bahrain_currency_model.keras and class_names.json.

---

## 🏷️ Supported Currency Classes

The classification system maps raw prediction indices to human-readable denomination labels:
- 5 Fils (Coin)
- 25 Fils (Coin)
- 50 Fils (Coin)
- 100 Fils (Coin)
- 0.500 BHD (Banknote)
- 1 BHD (Banknote)
- 5 BHD (Banknote)
- 10 BHD (Banknote)
- 20 BHD (Banknote)

---

## 💡 Practical Considerations & Future Scope

- **Background Impact:** Dense networks rely on spatial pixel feature maps; plain, contrasting backgrounds yield higher confidence scores.

- **Lighting Conditions:** Clear, non-reflective lighting helps distinguish fine details on coin faces and banknote patterns.

- **Future Work:** Replacing fully connected feedforward layers with Convolutional Neural Networks (CNNs) or Transfer Learning models (e.g., MobileNetV3 / EfficientNet) for stronger spatial translation invariance.

---

## 🔗 Links

* **Live Web App:** [Try the App on Streamlit](https://bhdcurrencyrecognition.streamlit.app/)

---
  
## 📁 Repository Structure

```text
bahraini-currency-recognition/
│
├── data/                              # All data assets
│   ├── bahrain_currency.zip           # Compressed augmented dataset (organized into subfolders per class)
│   └── pictures_mixed/                # Raw, unaugmented banknote/coin photos (front & back views)
│
├── notebooks/                         # All Jupyter / Colab notebooks
│   ├── GeneratingImages.ipynb         # Augmentation pipeline
│   └── Currency_Recognition_DL.ipynb  # Model training & evaluation
│
├── artifacts/                         # Exported model artifacts & metadata
│   ├── bahrain_currency_model.keras   # Saved Keras classification model
│   └── class_names.json               # Mapped class label metadata
│
├── app.py                             # Streamlit web interface
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
