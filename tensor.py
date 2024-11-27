import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'model.h5')

@st.cache_resource
def load_scaler_and_features():
    with open(r'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Define the feature names used during training
    feature_names = [
        "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model year", "origin"
    ]
    return scaler, feature_names

@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Janath\Mini Project\New folder\models\auto-mpg.csv')

# Load resources
model = load_model()
scaler, feature_names = load_scaler_and_features()
data = load_data()

# Streamlit UI
st.title("Auto MPG Prediction App")
st.write("This app predicts the miles per gallon (MPG) of automobiles based on input features.")

# Display the dataset
st.subheader("Dataset Overview")
if st.checkbox("Show dataset"):
    st.write(data)

# Input form
st.subheader("Enter Input Features")
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Predict button
if st.button("Predict"):
    try:
        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        # Scale the input features
        inputs_scaled = scaler.transform(input_df)
        
        # Reshape input for the model (1 batch, 1 timestep, 7 features)
        inputs_scaled = np.reshape(inputs_scaled, (1, 1, inputs_scaled.shape[1]))
        
        # Make predictions
        prediction = model.predict(inputs_scaled)
        
        # Extract scalar value from NumPy array
        prediction_value = float(prediction[0][0])  # Convert to Python float
        
        # Display the prediction
        st.success(f"Predicted MPG: {prediction_value:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

