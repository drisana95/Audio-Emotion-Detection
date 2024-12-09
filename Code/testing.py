import librosa
import numpy as np
import tensorflow as tf
import joblib  # Import joblib to load the LabelEncoder

# Load the saved model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Feature extraction from audio
def extract_features(audio_data):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Average across time
    return mfcc

# Predict emotion from a new audio file
def predict_emotion(model, label_encoder, audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    features = extract_features(audio)
    features = features.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)
    predicted_label_index = np.argmax(prediction)
    print(f"Predicted label index: {predicted_label_index}")  # Debugging line
    predicted_label = label_encoder.inverse_transform([predicted_label_index])
    return predicted_label

# Main execution
if __name__ == "__main__":
    model_path = r"C:\Users\drisa\Downloads\audio_emotion_model.h5"  # Path to your saved model
    audio_path = r"C:\Users\drisa\Downloads\archive (2)\Actor_24\03-01-05-02-01-02-24.wav"  # Path to the audio file for testing

    # Load the model
    model = load_model(model_path)

    # Load the label encoder
    label_encoder = joblib.load(r"C:\Users\drisa\Downloads\label_encoder.pkl")  # Load the saved label encoder

    # Predict emotion
    predicted_emotion = predict_emotion(model, label_encoder, audio_path)
    print(f'Predicted Emotion: {predicted_emotion[0]}')
