import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

# Load the trained ANN model
model = tf.keras.models.load_model('model/breast_cancer_model.keras')

# Load the scaler
with open('model/breast_cancer_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Input features (using st.number_input for numerical features)
selected_feature_names = ['mean radius', 'mean texture',  'mean perimeter', 'mean area', 
                          'mean smoothness', 'mean compactness', 'mean concavity', 
                          'mean concave points', 'mean symmetry', 'mean fractal dimension']

# Feature min-max values from df.describe()
feature_min_max = {
    'mean radius': (6.981000, 28.110000),
    'mean texture': (9.710000, 39.280000),
    'mean perimeter': (43.790000, 188.500000),
    'mean area': (143.500000, 2501.000000),
    'mean smoothness': (0.052630, 0.163400),
    'mean compactness': (0.019380, 0.345400),
    'mean concavity': (0.000000, 0.426800),
    'mean concave points': (0.000000, 0.201200),
    'mean symmetry': (0.106000, 0.304000),
    'mean fractal dimension': (0.049960, 0.097440)
}

# Streamlit app title
st.title("Breast Cancer Detection App")

# Input features with min-max validation
input_features = []
for feature in selected_feature_names:
    min_val, max_val = feature_min_max[feature]
    input_features.append(st.number_input(feature, min_value=min_val, max_value=max_val, value=min_val))


# Create a DataFrame from the input features
input_df = pd.DataFrame([input_features], columns=selected_feature_names)

# Preprocess the input data (feature scaling)
input_df_scaled = scaler.transform(input_df)

# Make prediction using the loaded model
prediction = model.predict(input_df_scaled)

# Display the prediction
st.subheader("Prediction:")
if prediction[0][0] >= 0.5:
    st.write("Benign")
else:
    st.write("Malignant")