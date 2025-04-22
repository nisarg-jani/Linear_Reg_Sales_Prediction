import streamlit as st
import pandas as pd
import pickle

# Load the pickled model and preprocessing objects
with open('model_pipeline.pkl', 'rb') as f:
    model_data = pickle.load(f)

lower_bound = model_data['lower_bound']
upper_bound = model_data['upper_bound']
scaler = model_data['scaler']
model = model_data['model']

# Static feature names
feature_names = ['TV', 'Radio', 'Newspaper']

# Streamlit app
st.title("Sales Prediction App")
st.write("Enter the advertising spend amounts in dollars ($) to predict Sales.")

# Input fields
tv = st.number_input("TV advertising spend ($)", value=0.0, format="%.2f")
radio = st.number_input("Radio advertising spend ($)", value=0.0, format="%.2f")
newspaper = st.number_input("Newspaper advertising spend ($)", value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    # Convert inputs from dollars to thousands of dollars
    input_data = [[tv/1000, radio/1000, newspaper/1000]]
    
    # Create input DataFrame
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Apply capping to Newspaper
    input_df['Newspaper'] = input_df['Newspaper'].clip(lower_bound, upper_bound)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display prediction
    st.success(f"Predicted Sales: {prediction[0]:.2f} Millions $")