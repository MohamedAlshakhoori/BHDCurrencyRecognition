# 🇧🇭 Bahraini Currency Recognition System

A Deep Learning computer vision system developed to classify Bahraini currency banknotes and coins across various denominations. Built using TensorFlow/Keras and deployed via an interactive Streamlit application supporting image uploads and live webcam captures.

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
