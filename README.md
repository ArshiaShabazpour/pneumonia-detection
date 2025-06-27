
---
# 🫁 Anomaly Detection in Chest X-Ray Images Using ConvLSTM Autoencoders

**Authors:**

Alireza Shahbazpourtazehkand 

Arshiya Shahbazpourtazehkand 

---

## 📌 Project Summary

This project demonstrates how **ConvLSTM-based autoencoders** can be used for **anomaly detection** in **chest X-ray (CXR)** images. The method focuses on detecting pneumonia by training on healthy images and flagging any images with high reconstruction error as potential anomalies.

---

## 🧠 Methodology

### 🛠️ Architecture

* **ConvLSTM Autoencoder**: Combines convolutional layers with LSTM to extract both spatial and temporal patterns (time dimension simulated as a single-frame video).

### 🧪 Training Strategy

* **Train only on healthy (NORMAL) CXR images**
* Use **reconstruction error** to detect anomalies in unseen test data

### 🖼️ Preprocessing

* Images are:

  * Converted to grayscale
  * Resized to **64x64**
  * Normalized to \[0, 1] range
  * Given an artificial **time dimension**

---

## 📂 Dataset

**Kaggle Dataset**: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paulti/chest-xray-images)
Folder structure expected:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

---

## 🚀 How to Run

### 📌 1. Train the Model

```bash
python convlstmauto.py
```

This will:

* Load CXR images from `chest_xray/NORMAL`
* Train a ConvLSTM autoencoder
* Save the model to:

  ```
  convlstm_autoencoder_model.h5
  ```

### 📌 2. Evaluate the Model

Make sure the test images are stored in:

```
chest_xray/test/NORMAL
chest_xray/test/PNEUMONIA
```

```bash
python ConvLSTMacc.py
```

This script:

* Loads the saved model
* Computes reconstruction errors on test images
* Flags anomalies using a threshold
* Prints classification accuracy, confusion matrix, and plots error histogram

---

## ✅ Example Output

```
Accuracy: 0.9756
Confusion Matrix:
[[117   1]
 [  5 385]]
Classification Report:
              precision    recall  f1-score   support
      NORMAL       0.96      0.99      0.97       118
   PNEUMONIA       1.00      0.99      0.99       390
```

---

## 🧪 Evaluation Details

* Reconstruction Error: Mean Absolute Error (MAE)
* Threshold Example:

  ```python
  threshold = 0.0000001  # You may need to tune this
  ```
* Prediction:

  ```python
  y_pred = (reconstruction_errors > threshold).astype(int)
  ```

---

## 📚 References

1. Raja Shenbagam et al., 2021 — Pneumonia detection with CNNs
2. Dastider et al., 2021 — Hybrid CNN-LSTM for COVID-19
3. Zhong Hong, 2024 — LSTM Autoencoders for Anomaly Detection
4. Alexandre Xavier, 2019 — Intro to ConvLSTM
5. Ankayarkanni et al., 2024 — Autoencoder-BiLSTM for Lung Disease Classification

---

## 📄 License

This project is licensed under the **MIT License**.

---
