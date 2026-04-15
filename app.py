import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Harga Mobil", page_icon="🚗")
st.title("🚗 Aplikasi Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil di bawah ini untuk mengetahui estimasi harga jualnya di pasar.")

# 2. Fungsi untuk memuat semua alat preprocessing & model
@st.cache_resource
def load_components():
    model = joblib.load('used_car_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_trans = joblib.load('le_trans.pkl')
    le_fuel = joblib.load('le_fuel.pkl')
    bm_mean = joblib.load('bm_mean_dict.pkl')
    return model, scaler, le_trans, le_fuel, bm_mean

model, scaler, le_trans, le_fuel, bm_mean = load_components()

# 3. Form Input Pengguna
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.text_input("Merek Mobil", value="Ford")
        model_car = st.text_input("Model Mobil", value="Mustang")
        model_year = st.number_input("Tahun Pembuatan", min_value=1990, max_value=datetime.now().year, value=2019)
        milage = st.number_input("Jarak Tempuh (Mil)", min_value=0, value=35000)
        
    with col2:
        engine = st.text_input("Spesifikasi Mesin", value="300.0HP 2.3L I4 Cylinder Engine")
        # Menggunakan class yang sudah disimpan encoder
        transmission = st.selectbox("Transmisi", le_trans.classes_)
        fuel_type = st.selectbox("Tipe Bahan Bakar", le_fuel.classes_)
        
    submit_button = st.form_submit_button(label="🔮 Prediksi Harga")

# 4. Proses Prediksi setelah tombol ditekan
if submit_button:
    # A. Preprocessing Umur & Milage
    current_year = datetime.now().year
    car_age = current_year - model_year
    car_age = 1 if car_age == 0 else car_age
    milage_per_year = milage / car_age
    
    # B. Preprocessing Kategori
    brand_model = brand + '_' + model_car
    
    # Jika brand_model baru tidak ada di data lama, gunakan nilai median
    median_price = pd.Series(bm_mean).median()
    bm_encoded = bm_mean.get(brand_model, median_price)
    
    trans_encoded = le_trans.transform([transmission])[0]
    fuel_encoded = le_fuel.transform([fuel_type])[0]
    
    # C. Ekstraksi Mesin (HP & Volume)
    hp_match = re.search(r'(\d+\.?\d*)HP', engine)
    hp = float(hp_match.group(1)) if hp_match else 200.0 # Nilai default aman
    
    vol_match = re.search(r'(\d+\.?\d*)L', engine)
    vol = float(vol_match.group(1)) if vol_match else 2.5 # Nilai default aman
    
    # D. Menyusun input untuk model
    input_df = pd.DataFrame([[
        bm_encoded, car_age, milage, milage_per_year,
        hp, vol, trans_encoded, fuel_encoded
    ]], columns=[
        'brand_model_encoded', 'car_age', 'milage', 'milage_per_year',
        'horsepower', 'engine_volume', 'transmission_encoded', 'fuel_encoded'
    ])
    
    # E. Scaling dan Prediksi
    try:
        input_scaled = scaler.transform(input_df)
        harga_prediksi = model.predict(input_scaled)[0]
        
        st.success("Berhasil melakukan prediksi!")
        st.metric(label="💰 Estimasi Harga Jual", value=f"${harga_prediksi:,.2f}")
        st.balloons() # Efek animasi balon saat berhasil
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")