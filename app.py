import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset (you can replace this with your own dataset)
df = pd.read_csv('src/Medical_data.csv')  # Replace with your actual file

# Check if the dataset has 'symptoms', 'diagnosis', and 'treatment' columns
if not all(col in df.columns for col in ['symptoms', 'diagnosis', 'treatment']):
    st.error("Dataset should contain 'symptoms', 'diagnosis', and 'treatment' columns.")
    st.stop()

# Preprocess the dataset
symptoms = df['symptoms']  # Assuming 'symptoms' column contains symptom information
diagnosis = df['diagnosis']  # Assuming 'diagnosis' column contains diagnoses

# Encode diagnosis labels
label_encoder = LabelEncoder()
diagnosis_encoded = label_encoder.fit_transform(diagnosis)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms, diagnosis_encoded, test_size=0.2, random_state=42)

# Convert symptoms (text) into numerical format (vectorize)
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train_vect.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(np.unique(diagnosis_encoded)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
with st.spinner("Training the model..."):
    model.fit(X_train_vect, y_train, epochs=10, batch_size=32, validation_data=(X_test_vect, y_test))

# Streamlit app
st.title("Healthcare Symptoms Checker")

# Input symptoms from the user
user_input = st.text_input("Enter your symptoms separated by commas (e.g. headache, fever, nausea):")

# Predict diagnosis and treatment
if st.button("Check Diagnosis"):
    if user_input:
        # Vectorize user input symptoms
        input_vect = vectorizer.transform([user_input]).toarray()
        
        # Predict diagnosis
        prediction = model.predict(input_vect)
        predicted_diagnosis = label_encoder.inverse_transform([np.argmax(prediction)])
        
        # Retrieve recommended treatment based on predicted diagnosis
        recommended_treatment = df[df['diagnosis'] == predicted_diagnosis[0]]['treatment'].values[0]
        
        # Display prediction and treatment
        st.subheader("Diagnosis Result")
        st.write(f"**Predicted Diagnosis:** {predicted_diagnosis[0]}")
        st.write(f"**Recommended Treatment:** {recommended_treatment}")
    else:
        st.write("Please enter symptoms.")
