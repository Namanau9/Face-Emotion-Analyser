# 😊 Real-Time Face Emotion Detection Web App

A Deep Learning based Face Emotion Detection system built using **CNN + OpenCV + Streamlit**.  
This project detects human emotions in real-time using a webcam through a web interface.

---

## 🚀 Features

- 🎥 Real-time webcam emotion detection
- 😊 Detects 7 emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- 📊 Confidence score display
- 🌐 Streamlit web interface
- 🧠 CNN trained on FER2013 dataset

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- streamlit-webrtc
- NumPy

---

## 📂 Project Structure

```
FaceEmotionDetection/
│
├── emotion_model.h5
├── train.py
├── app_webcam.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Namanau9/Face-Emotion-Analyser
```

### 2️⃣ Navigate into the Project Folder

```bash
cd FaceEmotionDetection
```

---

### 3️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

If you have a requirements.txt file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow opencv-python streamlit streamlit-webrtc numpy av
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app_webcam.py
```

Then open your browser and go to:

```
http://localhost:8501
```

Allow camera access and start detecting emotions 🎉

---

## 🧠 Model Details

- Dataset: FER2013
- Image Size: 48x48 (grayscale)
- Model Type: Convolutional Neural Network (CNN)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Output: 7 emotion classes

---

## 📊 How It Works

1. Webcam captures real-time video
2. Haar Cascade detects faces
3. Face is preprocessed (grayscale → resize → normalize)
4. CNN predicts emotion
5. Emotion label + confidence % shown on screen

---

## 📌 Future Improvements

- Transfer Learning (MobileNet / EfficientNet)
- Multimodal Emotion Detection (Voice + Face)
- Cloud Deployment (Streamlit Cloud / Render)
- Emotion Analytics Dashboard

---

## 👨‍💻 Author

Naman A U

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!
