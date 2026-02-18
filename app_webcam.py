import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- UI SETTINGS -----------------
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("😊 Real-Time Face Emotion Detection")
st.write("Live webcam emotion detection using Deep Learning")

# ---------------- LOAD MODEL ----------------
model = load_model("emotion_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ---------------- WEBCAM PROCESSING ----------------
class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face / 255.0
            face = face.reshape(1,48,48,1)

            prediction = model.predict(face)
            max_index = np.argmax(prediction[0])
            emotion = emotion_labels[max_index]
            confidence = round(float(np.max(prediction)) * 100, 2)

            label = f"{emotion} ({confidence}%)"

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,255,0),2)

        return img

# ---------------- START WEBCAM ----------------
webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
