# EyeQ: Vision Health Suite

A comprehensive web-based application built with Streamlit to support basic vision screening and awareness. The app offers tools for **Diabetic Retinopathy Classification** and multiple **Vision Tests** that simulate clinical assessments in a simplified format.

[🌐 Live App](https://vjmcenyiwjvqijuit6dbmc.streamlit.app/)

## 🚀 Features

### 🩺 Diabetic Retinopathy Classification

Upload a retinal image and receive an AI-driven classification of potential diabetic retinopathy stages:

* **No DR**
* **Mild DR**
* **Moderate DR**
* **Severe DR**
* **Proliferative DR**

Each classification includes an explanation and general educational content on diabetic retinopathy.

### 👁️ Vision Tests

* 🎨 **Color Vision Test** – Detect color blindness and deficiencies
* 👓 **Visual Acuity Test** – Measure clarity of vision
* ◐ **Contrast Sensitivity Test** – Assess ability to distinguish subtle shades
* 🔆 **Astigmatism Test** – Identify possible distortions in vision
* 👀 **Peripheral Vision Test** – Evaluate side vision capabilities
* ⚡ **Reaction Time Test** – Test response speed to visual cues
* 🔲 **Amsler Grid Test** – Screen for macular degeneration

---

## 📦 Installation

To run this app locally:

```bash
git clone https://github.com/yourusername/vision-health-suite.git
cd vision-health-suite
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 System Overview

### Diabetic Retinopathy Detection

This system leverages a convolutional neural network (CNN) trained on labeled retinal fundus images to classify the severity of diabetic retinopathy (DR).

### Dataset Description

* **Source:** Retinal fundus images labeled with DR severity (grades 0–4)
* **Structure:**

  * `test/0/` to `test/4/` folders by class
  * JPEG format
* **Preprocessing:**

  * Image resizing and normalization
  * Data augmentation: rotation, flipping, brightness/contrast adjustment

### Model Architecture

* **Framework:** PyTorch
* **Model Type:** Convolutional Neural Network (CNN)
* **Architecture:**

  * Convolutional layers with ReLU
  * MaxPooling
  * Dense (fully connected) layers
  * Final softmax layer for 5-class classification
* **Loss Function:** Cross-Entropy
* **Optimizer:** Adam

Trained model saved at: `model/final_dr_model.pt`

### Training Pipeline

* **Environment:** Python 3.x, PyTorch
* **Steps:**

  1. Load and preprocess dataset
  2. Train/validation/test split
  3. CNN training with early stopping
  4. Save best performing model

```python
from model import load_model, predict_image

model = load_model('model/final_dr_model.pt')
result = predict_image(model, 'path/to/image.jpeg')
print(f"Predicted DR grade: {result}")
```

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score (per class)
* Confusion Matrix (visualized)

*Insert actual results, graphs, or confusion matrix here if available.*

---

## 🔧 Future Enhancements

* Add support for real-time webcam capture
* Explore advanced architectures (EfficientNet, Vision Transformers)
* Add explainability tools like Grad-CAM for transparency
* Extend to multi-disease detection using fundus images
* Mobile and PWA version of the application

---

## 📚 References

1. [Kaggle Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
