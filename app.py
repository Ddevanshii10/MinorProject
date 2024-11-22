from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import sys
import os
from keras.models import load_model
# sys.setdefaultencoding('utf-8')

app = Flask(__name__)

# Load the LSTM model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'best_lstm_model.h5')

lstm_model = load_model(model_path)

# lstm_model = load_model('models/best_lstm_model.h5')

# Load and preprocess the dataset for label encoding
dataset = pd.read_csv('dataset.csv',encoding='utf-8')
label_encoder = LabelEncoder()
dataset['Accident_Severity'] = label_encoder.fit_transform(dataset['Accident_Severity'])

# Preprocess input data function
def preprocess_input(data):
    # Create DataFrame for processing
    df = pd.DataFrame(data, index=[0])

    # Fill missing values for categorical features with mode
    df['Road_Surface_Conditions'] = df['Road_Surface_Conditions'].fillna(df['Road_Surface_Conditions'].mode()[0])
    df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
    df['Weather_Conditions'] = df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0])
    
    # Fill missing values for numerical features with mean
    df['Longitude'] = df['Longitude'].fillna(df['Longitude'].mean())
    df['Number_of_Casualties'] = df['Number_of_Casualties'].fillna(df['Number_of_Casualties'].mean())
    df['Number_of_Vehicles'] = df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].mean())

    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ['Latitude', 'Longitude', 'Number_of_Casualties', 'Number_of_Vehicles']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # One-hot encode categorical features (if needed)
    df = pd.get_dummies(df, columns=['Road_Surface_Conditions', 'Road_Type', 'Weather_Conditions'], drop_first=True)

    # Prepare data for LSTM input (reshape for LSTM)
    scaled_data = df.values.reshape((1, -1, 1))  # Reshape to (1, seq_length, num_features)
    
    return scaled_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'Latitude': float(request.form['latitude']),
        'Longitude': float(request.form['longitude']),
        'Number_of_Casualties': float(request.form['number_of_casualties']),
        'Number_of_Vehicles': float(request.form['number_of_vehicles']),
        'Road_Surface_Conditions': request.form['road_surface_conditions'],
        'Road_Type': request.form['road_type'],
        'Weather_Conditions': request.form['weather_conditions']
    }

    # Preprocess input data
    processed_data = preprocess_input(input_data)

    try:
        # Make prediction using LSTM model
        lstm_prediction = lstm_model.predict(processed_data)

        # Convert prediction to accident severity label
        predicted_severity_index = np.argmax(lstm_prediction)  # Assuming a multi-class output from LSTM
        predicted_severity = label_encoder.inverse_transform([predicted_severity_index])[0]

        return render_template('index.html', lstm_result=predicted_severity)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run()
