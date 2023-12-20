import streamlit as st
import numpy as np
from model import load_model
import joblib

# Load model and label encoder
loaded_model, label_encoder = load_model()
label_encoder = joblib.load('label_encoder.pkl')

def predict_price(property_type, city, baths, bedrooms, area_type, area_size):
    try:
        property_type_encoded = label_encoder.transform([str(property_type)])[0]
        city_encoded = label_encoder.transform([str(city)])[0]
        area_type_encoded = label_encoder.transform([str(area_type)])[0]
    except ValueError as e:
        new_labels = [label for label in [property_type, city, area_type] if label not in label_encoder.classes_]
        label_encoder.classes_ = np.concatenate([label_encoder.classes_, new_labels])
        property_type_encoded = label_encoder.transform([str(property_type)])[0]
        city_encoded = label_encoder.transform([str(city)])[0]
        area_type_encoded = label_encoder.transform([str(area_type)])[0]

    input_array = np.array([property_type_encoded, city_encoded, baths, bedrooms, area_type_encoded, area_size]).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    predicted_price = f'Rs {prediction[0]:,.2f}'
    return predicted_price

def main():
    st.title('Real Estate Price Prediction')
    property_type = st.text_input('Property Type')
    city = st.text_input('City')
    baths = st.number_input('Number of Baths', min_value=1)
    bedrooms = st.number_input('Number of Bedrooms', min_value=1)
    area_type = st.text_input('Area Type')
    area_size = st.number_input('Area Size')

    if st.button('Predict'):
        try:
            result = predict_price(property_type, city, baths, bedrooms, area_type, area_size)
            st.success(f'Predicted Price: {result}')
        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()
