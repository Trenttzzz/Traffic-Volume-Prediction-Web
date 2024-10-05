import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Untuk memuat model pickle
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# Muat encoder dari file pickle
with open('encoder/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Fungsi untuk memproses data
# Fungsi untuk memproses data
def process_input_data(holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time):
    # Membuat DataFrame dari input
    data = pd.DataFrame({
        'holiday': [holiday],
        'temp': [temp],
        'rain_1h': [rain_1h],
        'snow_1h': [snow_1h],
        'clouds_all': [clouds_all],
        'weather_main': [weather_main],
        'weather_description': [weather_description],
        'date_time': [pd.to_datetime(date_time, format='%d-%m-%Y %H:%M')]
    })

    # Ekstraksi fitur waktu
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek  # 0=Senin, 6=Minggu
    data['day_of_month'] = data['date_time'].dt.day
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    data['is_weekend'] = data['date_time'].dt.weekday >= 5  # Sabtu dan Minggu

    # Mengubah Satuan Suhu dari Kelvin Menjadi Celcius
    data['temp_celcius'] = data['temp'] - 273.15

    # Kategori pada Kolom 'clouds_all'
    bins = [0, 30, 70, 100]
    labels = ['low', 'medium', 'high']
    data['cloud_categories'] = pd.cut(data['clouds_all'], bins=bins, labels=labels, include_lowest=True)

    # Menggabungkan holiday dengan is_weekend
    data['is_weekend_and_holiday'] = (data['is_weekend'] & data['holiday'].notnull())

    # Buat fitur 'is_precipitation'
    data['is_precipitation'] = (data['rain_1h'] > 0) | (data['snow_1h'] > 0)

    # Encoding weather_main menggunakan OneHotEncoder
    weather_main_encoded = encoder.transform(data[['weather_main']]).toarray()

    # Dapatkan nama kolom hasil encoding
    weather_main_encoded_df = pd.DataFrame(weather_main_encoded, columns=encoder.get_feature_names_out(['weather_main']))

    # Gabungkan hasil one-hot encoding ke DataFrame data
    data = pd.concat([data, weather_main_encoded_df], axis=1)

    # Encoding Categorical Data
    frequency = data['weather_description'].value_counts()
    data['weather_description_encoded'] = data['weather_description'].map(frequency)

    data['cloud_categories'] = pd.Categorical(data['cloud_categories'], categories=['low', 'medium', 'high'], ordered=True)
    data['cloud_categories_encoded'] = data['cloud_categories'].cat.codes

    # Mengisi nilai yang hilang untuk holiday
    data['holiday'] = data['holiday'].fillna('No Holiday')
    frequency = data['holiday'].value_counts()
    data['holiday_encoded'] = data['holiday'].map(frequency)

    # Handle missing value dari traffic_volumelag1, traffic_volumelag24, traffic_volumelag168
    for col in ['traffic_volume_lag1', 'traffic_volume_lag24', 'traffic_volume_lag168']:
        data[col] = 0

    # Menghapus kolom yang tidak diperlukan atau 
    data_cleaned = data.drop(columns=['date_time', 'cloud_categories', 'holiday', 'weather_description', 'weather_main'])

    return data_cleaned


# Fungsi untuk memuat model dari file pickle

def load_model():
    model_path = 'model/xgb_regressor_model.pkl'  # Path lengkap ke model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)  # Memuat model dari file
    return model

# Streamlit App
st.title("Input Data for Traffic Volume Prediction")

# Input fields for user data
holiday = st.text_input("Holiday (e.g. National Holiday, No Holiday)", "")
temp = st.number_input("Temperature (in Kelvin)", min_value=0.0)
rain_1h = st.number_input("Rain in last hour (0 or 1)", min_value=0, max_value=1, step=1)
snow_1h = st.number_input("Snow in last hour (0 or 1)", min_value=0, max_value=1, step=1)
clouds_all = st.number_input("Cloud Cover Percentage (0 - 100)", min_value=0, max_value=100)
weather_main = st.text_input("Weather Main (e.g. Clouds, Rain, Clear)", "")
weather_description = st.text_input("Weather Description (e.g. light rain, overcast clouds)", "")
date_time = st.text_input("Date and Time (format: dd-mm-yyyy HH:MM)", "")

# Button to process input
if st.button("Process Data"):
    try:
        processed_data = process_input_data(holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time)
        st.write("Data Processed Successfully")

        st.write("Processed Data Shape:", processed_data.shape)
        st.write("Processed Data Shape:", processed_data.columns)

        # Load model and make prediction
        model = load_model()
        prediction = model.predict(processed_data)

        st.write(f"Predicted Traffic Volume: {prediction[0]} vehicles")


    except Exception as e:
        st.error(f"An error occurred: {e}")
