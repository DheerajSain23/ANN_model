import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# ================================
# Load Trained Model
# ================================
model = tf.keras.models.load_model('model.h5')

# ================================
# Load Saved Preprocessing Objects
# ================================
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# ================================
# Streamlit App Title
# ================================
st.title("Customer Churn Prediction App")

# ================================
# User Inputs
# ================================
geography = st.selectbox(
    'Select Geography',
    onehot_encoder_geography.categories_[0]
)

gender = st.selectbox(
    'Select Gender',
    label_encoder_gender.classes_
)

credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=850
)

age = st.number_input(
    'Age',
    min_value=18,
    max_value=100
)

tenure = st.slider(
    'Tenure (Years with Bank)',
    min_value=0,
    max_value=10
)

balance = st.number_input(
    'Account Balance',
    min_value=0.0
)

num_of_products = st.slider(
    'Number of Products',
    min_value=1,
    max_value=4
)

has_cr_card = st.selectbox(
    'Has Credit Card?',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member?',
    [0, 1]
)

estimated_salary = st.number_input(
    'Estimated Salary',
    min_value=0.0
)

# ================================
# Create Input DataFrame
# ================================
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# ================================
# One-Hot Encode Geography Column
# ================================
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()

geo_columns = onehot_encoder_geography.get_feature_names_out(['Geography'])

geo_df = pd.DataFrame(
    geo_encoded,
    columns=geo_columns
)

# ================================
# Combine Original + Encoded Data
# ================================
input_final = pd.concat(
    [input_data, geo_df],
    axis=1
)

# ================================
# Reorder Columns to Match Training
# ================================
input_final = input_final[scaler.feature_names_in_]

# ================================
# Scale Input Data
# ================================
input_scaled = scaler.transform(input_final)

# ================================
# Predict Churn Probability
# ================================
prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]

# ================================
# Show Prediction Result
# ================================
st.subheader("Prediction Result")

st.write(f"Churn Probability: {prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is unlikely to churn.")