# Plant Disease Recognition System

A web-based Streamlit application powered by Deep Learning (TensorFlow/Keras CNN) to identify and classify plant diseases from leaf images. The system is trained on the Kaggle New Plant Diseases Dataset, covering 38 classes of healthy and diseased leaves across apples, potatoes, tomatoes, grapes, corn, and more.

Developed as a showcase portfolio application with a local mock fallback when model weights are not present.

---

## 🚀 Key Features

* **38 Disease Classes:** Supports robust leaf recognition for various crops.
* **Auto-Fallback Engine:** If the heavy 99MB pre-trained model file (`plant_disease_cnn_model.keras`) is missing, the app dynamically constructs and registers a lightweight CNN model at startup to ensure it runs out-of-the-box.
* **Streamlit UI:** Interactive web interface with a responsive leaf uploader, image previews, and prediction display.
* **Automatic Temp File Cleanup:** Ensures upload artifacts are deleted immediately after classification to keep storage clean.

---

## 📁 Project Structure

```text
Plant_disease_dectection_system/
├── app.py                      # Main Streamlit web application & prediction logic
├── PlantDiseaseDetection.ipynb # Jupyter notebook used to train the CNN model
├── requirements.txt            # Project python dependencies
├── Disease.png                 # Main homepage visual banner
├── README.md                   # Setup guide and documentation
└── test/                       # Collection of leaf test images (JPEG format)
```

---

## 💻 Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Kushagriisharma/Plant_disease_dectection_system.git
cd Plant_disease_dectection_system
```

### 2. Install Dependencies
It is highly recommended to run this inside a virtual environment (e.g. `venv` or `conda`):
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit server from your terminal:
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## 📊 Dataset Reference
The model is trained on the [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
