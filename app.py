from flask import Flask, render_template, request, jsonify
import numpy as np
from model import load_model
import joblib

app = Flask(__name__)

loaded_model, label_encoder = load_model()

label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        try:
            property_type = data['propertyType']
            city = data['city']
            baths = float(data['baths'])
            bedrooms = float(data['bedrooms'])
            area_type = data['areaType']
            area_size = float(data['areaSize'])

            print("Received data:", data)
            print("Received input:", property_type, city, baths, bedrooms, area_type, area_size)

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

            return jsonify({'prediction': predicted_price})

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
