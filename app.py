# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dummy dataset
def load_data():
    data = {
        'symptom_1': ['fever', 'cough', 'headache', 'fever', 'nausea'],
        'symptom_2': ['cough', 'headache', 'nausea', 'sore throat', 'fatigue'],
        'diagnosis': ['flu', 'cold', 'migraine', 'flu', 'food poisoning'],
        'treatment': ['rest and fluids', 'rest and fluids', 'painkillers', 'rest and fluids', 'hydration']
    }
    df = pd.DataFrame(data)
    return df

# Data preparation
df = load_data()
X = df[['symptom_1', 'symptom_2']].values
y = df['diagnosis'].values

# Encoding categorical data
symptom_encoder = LabelEncoder()
X[:, 0] = symptom_encoder.fit_transform(X[:, 0])
X[:, 1] = symptom_encoder.fit_transform(X[:, 1])

diagnosis_encoder = LabelEncoder()
y = diagnosis_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))  # Two input features (symptom_1 and symptom_2)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output for binary classification (adjust based on the number of classes)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

# Define a function to predict diagnosis based on symptoms
def predict_diagnosis(symptom_1, symptom_2):
    input_data = np.array([[symptom_1, symptom_2]])
    input_data[:, 0] = symptom_encoder.transform(input_data[:, 0])
    input_data[:, 1] = symptom_encoder.transform(input_data[:, 1])
    prediction = model.predict(input_data)
    diagnosis = diagnosis_encoder.inverse_transform([int(prediction[0][0])])
    return diagnosis[0]

# Streamlit app
def main():
    st.title('Healthcare Symptoms Checker')

    st.write("""
    This app uses a deep learning model to predict possible diagnoses based on your symptoms.
    Please enter your symptoms below.
    """)

    symptom_1 = st.selectbox('Select Symptom 1', df['symptom_1'].unique())
    symptom_2 = st.selectbox('Select Symptom 2', df['symptom_2'].unique())

    if st.button('Check Diagnosis'):
        diagnosis = predict_diagnosis(symptom_1, symptom_2)
        st.success(f'Predicted Diagnosis: {diagnosis}')

        # Optionally, show treatment based on the diagnosis
        treatment = df[df['diagnosis'] == diagnosis]['treatment'].values[0]
        st.write(f'Suggested Treatment: {treatment}')

if __name__ == '__main__':
    main()
