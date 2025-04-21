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
st.write("Enter the values for TV, Radio, and Newspaper advertising spend to predict Sales.")

# Input fields
tv = st.number_input("Enter TV advertising spend(In thousand $)", value=0.0, format="%.2f")
radio = st.number_input("Enter Radio advertising spend(In thousand $)", value=0.0, format="%.2f")
newspaper = st.number_input("Enter Newspaper advertising spend(In thousand $)", value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    # Create input DataFrame in the correct column order
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=feature_names)

    # Step 1: Apply capping to Newspaper
    input_df['Newspaper'] = input_df['Newspaper'].clip(lower_bound, upper_bound)

    # Step 2: Scale the input
    input_scaled = scaler.transform(input_df)

    # Step 3: Make prediction
    prediction = model.predict(input_scaled)

    # Display prediction
    st.success(f"Predicted Sales: {prediction[0]:.2f} Million $")
