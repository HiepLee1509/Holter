# .\venv\Scripts\activate
# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import ast

from src.backend import load_arrhythmia_model, get_model_input_length
from src.ui_config import setup_page_config, apply_custom_css
from src.view_single import render_single_analysis
from src.view_batch import render_batch_analysis

# 1. CẤU HÌNH TRANG & GIAO DIỆN
setup_page_config()
apply_custom_css()

# --- TẢI MODEL ---
@st.cache_resource
def setup_model(model_path):
    if not os.path.exists(model_path):
        return None, 0
    model = load_arrhythmia_model(model_path)
    input_len = get_model_input_length(model)
    return model, input_len

MODEL_PATH = "model\\ecg_model_code 17_t5.h5"
model, REQUIRED_LENGTH = setup_model(MODEL_PATH)

# SIDEBAR (INPUT & SETTINGS)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80)
    st.title("AI Heart Guard")
    st.markdown("---")
    
    st.header("1. Nhập Dữ Liệu")
    uploaded_file = st.file_uploader("Upload JSON/CSV", type=["json", "csv"])
    
    st.header("2. Tham số Kỹ thuật")
    fs = st.number_input("Tần số lấy mẫu (Hz)", 100, 1000, 360, help="MIT-BIH thường là 360Hz")
    
    with st.expander("Nâng cao (Wavelet/Peak)"):
        wavelet_type = st.selectbox("Wavelet Type", ['sym8', 'db4', 'db8'], index=0)
        r_peak_height = st.slider("Min Peak Height", 0.1, 5.0, 0.5)
    
    st.markdown("---")
    st.caption("Developed by Lê Nghĩa Hiệp\nMSSV: 20235326")

# MAIN CONTENT
st.title("🫀 Phân tích & Chẩn đoán Rối loạn nhịp tim ECG")
st.markdown("Hệ thống hỗ trợ chẩn đoán tự động sử dụng mô hình AI **Deep Learning (CNN + LSTM)**.")

# Model Checking
if model is None:
    st.error(f"⚠️ Không tìm thấy file model tại `{MODEL_PATH}`. Vui lòng kiểm tra lại thư mục dự án.")
    st.stop()

# Controller Logic
patient_data_map = {}

if uploaded_file is not None:
    try:
        # Read JSON/CSV file
        if uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                for i, d in enumerate(data):
                    patient_data_map[d.get("id", f"Rec {i}")] = d["reading"]
            elif isinstance(data, dict):
                patient_data_map = data
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if 'reading' in df.columns:
                 for i, row in df.iterrows():
                    val = row['reading']
                    reading = ast.literal_eval(val) if isinstance(val, str) else val
                    patient_data_map[str(row.get('id', f"Row {i}"))] = reading
            else:
                 for i in range(len(df)):
                    reading = df.iloc[i].values.tolist()
                    if len(reading)>100: patient_data_map[f"Row {i}"] = reading
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        
# ROUTING TO VIEWS
if patient_data_map:
    st.success(f"✅ Đã tải thành công dữ liệu {len(patient_data_map)} id của bệnh nhân.")
    
    # Create Tabs for Single and Batch Analysis
    tab_single, tab_batch = st.tabs(["👤 Phân tích trên 1 id cụ thể (Single mode)", "👥 Quét toàn bộ (Scan mode)"])

    with tab_single:
        # Call View Single
        render_single_analysis(
            patient_data_map, 
            model, 
            fs, 
            wavelet_type, 
            r_peak_height, 
            REQUIRED_LENGTH
        )

    with tab_batch:
        # Call View Batch
        render_batch_analysis(
            patient_data_map, 
            model, 
            fs, 
            wavelet_type, 
            r_peak_height
        )

else:
    st.info("👈 Vui lòng tải lên file dữ liệu (JSON hoặc CSV) ở thanh bên trái để bắt đầu.")