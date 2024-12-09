import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib  # Import joblib to save the LabelEncoder
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Step 1: Load the TESS dataset
def load_tess_data(dataset_path):
    data = []
    labels = []
    paths = []

    for foldername in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, foldername)
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                label = foldername  # Use the folder name as label
                audio, sr = librosa.load(file_path, sr=None)  # Load audio file
                data.append(audio)
                labels.append(label)
                paths.append(file_path)  # Store the file path

    return data, labels, paths

# Step 2: Feature extraction
def extract_features(audio_data):
    features = []
    
    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)  # Average across time
        features.append(mfcc)
        
    return np.array(features)

# Step 3: Build and train the model
def train_model(dataset_path):
    # Load the dataset
    data, labels, paths = load_tess_data(dataset_path)

    # Extract features
    X = extract_features(data)
    y = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Step 4: Build the model
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(13,)))  # Input shape for MFCCs
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Step 5: Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Step 6: Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Get the training accuracy from the last epoch
    training_accuracy = history.history['accuracy'][-1] * 100
    print(f'Training Accuracy after last epoch: {training_accuracy:.2f}%')

    # Save the model and the label encoder
    model.save(r"C:\Users\drisa\Downloads\audio_emotion_model.h5")
    joblib.dump(label_encoder, r"C:\Users\drisa\Downloads\label_encoder.pkl")  # Save the label encoder
    print("Model and label encoder saved.")

    # Step 7: Plot accuracy graph
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(r"C:\Users\drisa\Downloads\accuracy_graph.png")  # Save the accuracy graph
    plt.show()  # Display the graph

    return model, label_encoder, history

# Main execution
if __name__ == "__main__":
    dataset_path = r"C:\Users\drisa\Downloads\archive (1)\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data"  # Set your dataset path here
    model, label_encoder, history = train_model(dataset_path)
