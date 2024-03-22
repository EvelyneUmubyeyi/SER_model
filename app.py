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
import tempfile
import pickle
from st_audiorec import st_audiorec
import streamlit.components.v1 as components
from streamlit import session_state
import json

model = tf.keras.models.load_model('ser_model_african_optimized.h5')

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
    features_dataset = pd.DataFrame(features)
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
        if isinstance(uploaded_file, bytes):
            temp_file.write(audio_file)
        else:
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

def ChangeButtonColour(widget_label, font_color, background_color='transparent', x_padding='20px', y_padding='5px'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}';
                    elements[i].style.padding = '{y_padding} {x_padding}';
                    elements[i].style.transition = 'background-color 0.3s, color 0.3s';
                    elements[i].style.border = 'none';
                    elements[i].style.marginBottom = '0';
                    elements[i].addEventListener("mouseover", function() {{
                        this.style.opacity = '{'50%'}';
                    }});
                    elements[i].addEventListener("mouseout", function() {{
                        this.style.opacity = '{'100%'}';
                    }});
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


def get_prediction(file):
    temp_path = get_audio_data(file)
    features = get_audio_features(temp_path)
    X =[]
    for i in features:
        X.append(i)
    preprocessed_features = preprocessing(X)
    predicted_emotion = predict(preprocessed_features)
    return predicted_emotion

def load_lottie(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="EmotionDetect", page_icon="🎙", layout='wide')
with open('feedback.json', 'r') as file:
    emotions_data = json.load(file)

def display_emotion_info(emotion_name):
    pass

lottie_file = load_lottie("https://lottie.host/4d5b2501-51d4-4914-8905-85da70376358/ShmUpG2m0J.json")

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Hello :wave: How are you doing today😊')
        st.title('Welcome to EmotionDetect💝')
        st.write('''
            This innovative web app harnesses the power of Speech Emotion Recognition (SER) technology to decode emotions 
            from your voice. With EmotionDetect, you can effortlessly record your voice and instantly uncover the emotional 
            nuances within. 
                 
            You will gain insight into your emotions and receive basic mental health support to navigate them, alongside 
            access to a wealth of other resources on mental health available on the platform, as well as contacts of mental health 
            proffessional you can reach out to.
        ''')

    with right_column:
        if lottie_file is not None:
            st_lottie(lottie_file, height=500, speed=1, key="animated_image")
        else:
            st.write("Failed to load Lottie animation.")


with st.container():
    def upload_button_clicked():
        st.session_state.voice_space = 'upload'

    def record_button_clicked():
        st.session_state.voice_space = 'record'

    with st.container():  
        st.subheader('Record your self here')
        cols = st.columns(11)

        if 'voice_space' not in session_state:
            st.session_state.voice_space = 'record'

        if session_state.voice_space == 'record':
            uploaded_file = st_audiorec()
        elif session_state.voice_space == 'upload':
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
            st.audio(uploaded_file)
        cols[0].button('Record', key='b1', on_click=record_button_clicked)
        cols[1].button('Upload', key='b2', on_click=upload_button_clicked)

        if uploaded_file is not None:
            # if len(uploaded_file.getvalue()) > 512000:
            #     st.error("File size exceeds the limit. Please upload a file within 500KB.")
            # else:
            predicted_emotion = get_prediction(uploaded_file)
            with st.container():
                st.write('')
                if predicted_emotion == "sad":
                    st.subheader(f"😞{predicted_emotion.title()}")
                elif predicted_emotion == "happy":
                    st.subheader(f"😊{predicted_emotion.title()}")
                elif predicted_emotion == "fear":
                    st.subheader(f"😱{predicted_emotion.title()}")
                elif predicted_emotion == "disgust":
                    st.subheader(f"🤢{predicted_emotion.title()}")
                elif predicted_emotion == "neutral":
                    st.subheader(f"😐{predicted_emotion.title()}")
                else:
                    st.subheader(f"😠{predicted_emotion.title()}")

                st.markdown(f"<h4><b>✅Every emotion counts!</b></h4>", unsafe_allow_html=True)
                st.write(emotions_data[predicted_emotion]['validation_sentence'])
                st.markdown(f"<h4><b>💡Things you can know about feeling {predicted_emotion}</b></h4>", unsafe_allow_html=True)
                for fact in emotions_data[predicted_emotion]['facts']:
                    st.write("-", fact)
                if predicted_emotion == "sad":
                    st.markdown(f"<h4><b>📋Creative things you can try to cope with sadness</b></h4>", unsafe_allow_html=True)
                elif predicted_emotion == "happy":
                    st.markdown(f"<h4><b>📋Creative things you can try to celebrate happiness</b></h4>", unsafe_allow_html=True)
                elif predicted_emotion == "neutral":
                    st.markdown(f"<h4><b>📋Creative things you can try to embrace neutrality</b></h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h4><b>📋Creative things you can try to cope with {predicted_emotion}</b></h4>", unsafe_allow_html=True)

                for method in emotions_data[predicted_emotion]['coping_methods']:
                    st.write("-", method)
                
        ChangeButtonColour('Record', '#ffffff', '#f92f60') 
        ChangeButtonColour('Upload', '#ffffff', '#ffb02e') 