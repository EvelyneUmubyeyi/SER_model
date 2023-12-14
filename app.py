import numpy as np
import requests
import librosa
import tensorflow as tf
import streamlit as st
import pandas as pd
import soundfile as sf
from streamlit_lottie import st_lottie
from st_audiorec import st_audiorec
from sklearn.preprocessing import StandardScaler
from pydub.utils import mediainfo
import tempfile
import pickle


emotion_messages = {
    'angry': "Take a deep breath and count to ten. Try redirecting your energy into something positive.",
    'disgust': "Acknowledge your feelings, then focus on things that bring you joy and positivity.",
    'fear': "Remember, you're stronger than your fears. Try grounding techniques to center yourself.",
    'happy': "Embrace this happiness! Share it with loved ones or indulge in activities that amplify your joy.",
    'neutral': "Take a moment for yourself. Explore hobbies or activities that bring a sense of calmness.",
    'sad': "Allow yourself to feel, but also seek moments of comfort and activities that uplift your spirits."
}

model = tf.keras.models.load_model('ser_model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def predict(preprocessed_features):
    prediction = model.predict(preprocessed_features)
    emotion_index = np.argmax(prediction, axis=1)
    emotions_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    return emotions_list[int(emotion_index)]

def check_for_missing_values(arr):
    missing_values = np.isnan(arr)
    if np.any(missing_values):
        return True
    else:
        return False
    
def preprocessing(features):
    features_dataset = pd.DataFrame(X)
    features_dataset = features_dataset.fillna(0)
    features_dataset = features_dataset.values.reshape(1, -1)
    if features_dataset.shape[1] < 2376:
        num_missing_features = 2376 - features_dataset.shape[1]
        X_pred_padded = np.pad(features_dataset, ((0, 0), (0, num_missing_features)), mode='constant')
        scaled_X_pred = scaler.transform(X_pred_padded)
    else:
        scaled_X_pred = scaler.transform(features_dataset)
        
    scaled_X_pred=np.expand_dims(scaled_X_pred,axis=2)

    return scaled_X_pred


def get_audio_data(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_path = temp_file.name
        return temp_path
    

def extract_features(data, sr, frame_length=2048, hop_length=512):
  result = np.array([])

  # Zero Crossing Rate
  zcr=np.squeeze(librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length))

  # Root Mean Square Value
  rmse=np.squeeze(librosa.feature.rms(y = data,frame_length=frame_length,hop_length=hop_length))

  # Mel Frequency Cepstral Coefficients
  mfcc=np.ravel(librosa.feature.mfcc(y = data,sr=sr).T)

  result=np.hstack((result, zcr, rmse, mfcc))

  return result

def get_audio_features(path):
    sampling_data,sampling_rate = librosa.load(path, duration=2.5, offset=0.6)
    audio_features = extract_features(sampling_data, sampling_rate)
    features = np.array(audio_features)
    return features

def load_lottie(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="EmotionDetect", page_icon="🎙", layout='wide')


lottie_file = load_lottie("https://lottie.host/4d5b2501-51d4-4914-8905-85da70376358/ShmUpG2m0J.json")


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Hello :wave: How are you doing today😊')
        st.title('Welcome to EmotionDetect💝')
        st.write('''
This innovative web app harnesses the power of Speech Emotion Recognition (SER) technology to decode emotions from your voice. With EmotionDetect, you can effortlessly record your voice and instantly uncover the emotional nuances within. 

Experience the captivating world of AI-driven emotion detection and gain insight into the hidden dimensions of your speech. Discover, explore, and uncover the emotions within your voice with EmotionDetect – where your voice tells the story of your emotions.
''')

    with right_column:
        if lottie_file is not None:
            st_lottie(lottie_file, speed=1, height=400, key="animated_image")
        else:
            st.write("Failed to load Lottie animation.")


with st.container():
    st.write('---')
    st.write('''Record a short audio below about how you're felling your emotion will be detected!''')

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        if len(uploaded_file.getvalue()) > 512000:
            st.error("File size exceeds the limit. Please upload a file within 500KB.")
        else:
            temp_path = get_audio_data(uploaded_file)
            features = get_audio_features(temp_path)
            X =[]
            for i in features:
                X.append(i)
            preprocessed_features = preprocessing(X)
            emotion = predict(preprocessed_features)
            st.subheader(emotion.title())
            st.write(emotion_messages[emotion])
    