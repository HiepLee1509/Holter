import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
from src.backend import (
    denoise_signal_wavelet, 
    detect_and_segment, 
    predict_from_segments, 
    calculate_hrv_metrics,
    CLASS_INFO
)

from src.ui_plots import (
    plot_interactive_ecg, 
    plot_classes_pie, 
    plot_beat_shape, 
    plot_poincare_chart
)

def render_single_analysis(patient_data_map, model, fs, wavelet_type, r_peak_height, required_length):
    """Hàm hiển thị giao diện phân tích từng ca"""
    
    # 1. Selector for ID
    selected_id = st.selectbox("Chọn id để bắt đầu phân tích:", list(patient_data_map.keys()))
    raw_ecg = np.array(patient_data_map[selected_id])
    
    # 2. Button Bắt đầu phân tích
    col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
    with col_act2:
        start_btn = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN AI", type="primary", use_container_width=True)

    # 3. Call backend processing when button clicked
    if start_btn:
        with st.status("Đang phân tích...", expanded=True) as status:
            st.write("🔹 Đang khử nhiễu tín hiệu (Wavelet Denoising)...")
            denoised = denoise_signal_wavelet(raw_ecg, wavelet=wavelet_type)
            
            st.write("🔹 Đang phát hiện đỉnh R và phân đoạn nhịp...")
            segments, peaks = detect_and_segment(denoised, r_peak_height, output_length=required_length)
            
            if len(segments) > 0:
                st.write("🔹 Đang chạy mô hình AI (CNN-LSTM)...")
                pred_codes, _ = predict_from_segments(segments, model)
                
                st.write("🔹 Đang phân tích chuyên sâu HRV & Poincaré...")
                hrv_metrics = calculate_hrv_metrics(peaks, fs=fs)

                # Lưu kết quả vào Session State
                st.session_state.single_result = {
                    "raw": raw_ecg, "denoised": denoised,
                    "segments": segments, "peaks": peaks,
                    "codes": pred_codes,
                    "hrv": hrv_metrics
                }
                status.update(label="✅ Phân tích hoàn tất!", state="complete", expanded=False)
            else:
                status.update(label="⚠️ Không tìm thấy nhịp tim!", state="error")
                st.error("Không tách được nhịp tim nào. Hãy chỉnh lại ngưỡng 'Min Peak Height'.")

    # 4. Show results if available - Dashboard + Tabs
    if 'single_result' in st.session_state:
        res = st.session_state.single_result
        
        # --- Metrics Dashboard ---
        total_beats = len(res['codes'])
        abnormal_beats = sum([1 for c in res['codes'] if c != 'N'])
        abnormal_rate = (abnormal_beats / total_beats) * 100
        
        # Calculate BPM
        if len(res['peaks']) > 1:
            avg_distance = np.mean(np.diff(res['peaks']))
            bpm = 60 / (avg_distance / fs)
        else:
            bpm = 0
            
        st.markdown("### 📊 Tổng quan sức khỏe tim mạch")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nhịp tim trung bình", f"{int(bpm)} BPM", delta=f"{int(bpm-75)}" if bpm>0 else None, delta_color="inverse")
        m2.metric("Tổng số nhịp đã quét", f"{total_beats}")
        m3.metric("Số nhịp bất thường", f"{abnormal_beats}", delta=f"-{abnormal_beats}" if abnormal_beats > 0 else "Tốt", delta_color="normal")
        m4.metric("Tỷ lệ bất thường", f"{abnormal_rate:.1f}%", delta_color="inverse")
        
        st.divider()

        # Tabs for detailed analysis
        tab_overview, tab_details, tab_hrv, tab_monitor, tab_data = st.tabs([
            "🔎 Biểu đồ & Chẩn đoán", 
            "💓 Soi chi tiết từng nhịp", 
            "❤️ HRV & Poincaré", 
            "📺 Real-time Monitor",
            "📋 Dữ liệu bảng"
        ])
        
        # TAB 1: OVERVIEW
        with tab_overview:
            st.subheader("📈 Biểu đồ ECG tương tác")
            fig = plot_interactive_ecg(res['raw'], res['peaks'], res['codes'], fs=fs)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📝 Kết luận & Lời khuyên từ bác sĩ AI")
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_pie = plot_classes_pie(res['codes'])
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                # Show summary advice
                counts = pd.Series(res['codes']).value_counts()
                for code in counts.index:
                    info = CLASS_INFO[code]
                    css_class = "safe" if code == 'N' else "danger" if code in ['V', 'F'] else "warning"
                    st.markdown(f"""
                    <div class="advice-card {css_class}">
                        <h4>{info['name']} (Nhãn tương ứng: {code}) - {counts[code]} lần</h4>
                        <p>{info['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 2: DETAILS
        with tab_details:
            st.info("Kéo thanh trượt bên dưới để xem hình thái sóng của từng nhịp đập.")
            beat_idx = st.slider("Chọn nhịp thứ:", 0, total_beats-1, 0)
            
            curr_segment = res['segments'][beat_idx]
            curr_code = res['codes'][beat_idx]
            
            col_d1, col_d2 = st.columns([3, 1])
            with col_d1:
                fig_beat = plot_beat_shape(curr_segment, curr_code, beat_idx+1)
                st.plotly_chart(fig_beat, use_container_width=True)
            with col_d2:
                info = CLASS_INFO[curr_code]
                st.markdown(f"""
                ### Kết quả:
                <h2 style='color:{info['color']}'>{info['name']}</h2>
                """, unsafe_allow_html=True)

        # TAB 3: HRV
        with tab_hrv:
            hrv = res.get('hrv')
            if hrv is None:
                st.warning("⚠️ Không đủ dữ liệu đỉnh R để phân tích biến thiên nhịp tim (cần ít nhất 2 nhịp).")
            else:
                st.subheader("Phân tích Biến thiên nhịp tim (Heart Rate Variability)")
                
                # 1. Show SDNN & RMSSD
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.metric(
                        label="SDNN (Độ lệch chuẩn RR)",
                        value=f"{hrv['sdnn']:.2f} ms",
                        help="SDNN < 50ms: Sức khỏe kém/Nguy cơ cao. SDNN > 100ms: Tim khỏe mạnh."
                    )
                    st.info("""
                    **SDNN (Standard Deviation of NN intervals):** Phản ánh sức khỏe tổng quát của hệ tim mạch. Giá trị càng cao cho thấy khả năng thích ứng của tim càng tốt trước stress.
                    """)
                    
                with col_h2:
                    st.metric(
                        label="RMSSD (Căn bậc 2 trung bình hiệu số)",
                        value=f"{hrv['rmssd']:.2f} ms",
                        help="RMSSD thấp liên quan đến stress, mệt mỏi hoặc bệnh lý."
                    )
                    st.info("""
                    **RMSSD (Root Mean Square of Successive Differences):**
                    Phản ánh hoạt động của hệ thần kinh phó giao cảm. Dùng để đánh giá mức độ phục hồi của cơ thể.
                    """)

                st.divider()
                
                c_plot, c_text = st.columns([2, 1])
                with c_plot:
                    fig_poincare = plot_poincare_chart(hrv)
                    st.plotly_chart(fig_poincare, use_container_width=True)
                with c_text:
                        st.markdown("""
                        ### 🩺 Cách đọc biểu đồ:
                        
                        **1. Hình dáng "Cây vợt" (Tennis Racket) 🎾:**
                        * Các điểm tập trung thành cụm hình bầu dục dọc theo đường chéo.
                        * 👉 **Ý nghĩa:** Tim hoạt động ổn định, khỏe mạnh (Sinus Rhythm).
                        
                        **2. Hình "Quạt" hoặc phân tán (Fan/Complex) 🌪️:**
                        * Các điểm toả rộng ra xa đường chéo.
                        * 👉 **Ý nghĩa:** Dấu hiệu của Rung nhĩ (AFib) hoặc suy tim sung huyết.
                        
                        **3. Các cụm rời rạc (Islands):**
                        * Có các cụm điểm tách biệt hẳn so với đám đông chính.
                        * 👉 **Ý nghĩa:** Dấu hiệu của Ngoại tâm thu (SVEB/VEB) xen kẽ nhịp thường.
                        """)

        # TAB 4: MONITOR (Real-time Simulation)
        with tab_monitor:
            render_monitor_tab(res)

        # TAB 5: DATA TABLE
        with tab_data:
            df_export = pd.DataFrame({
                "Nhịp thứ": range(1, total_beats+1),
                "Thời gian (s)": res['peaks'] / fs,
                "Loại nhịp": [CLASS_INFO[c]['name'] for c in res['codes']],
                "Mã": res['codes']
            })
            st.dataframe(df_export, use_container_width=True)
            
            # CSV Download
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải xuống báo cáo CSV",
                csv,
                "ecg_analysis_report.csv",
                "text/csv",
                key='download-csv'
            )
        

def render_monitor_tab(res):
    """Tách logic Monitor ra hàm con cho gọn"""
    st.header("🏥 ICU Monitor Simulator")
    st.caption("Mô phỏng màn hình theo dõi sinh hiệu thời gian thực.")

    # 1. Pre-processing
    # 'Normal' (Green) và 'Danger' (Red)
    if 'monitor_data' not in st.session_state:
        # Ensure data is in NumPy array
        full_signal = np.array(res['denoised']).flatten()
        
        # Create two separate arrays for Normal and Danger
        normal_signal = full_signal.copy()
        danger_signal = np.full(full_signal.shape, np.nan)

        # Find indices of abnormal beats
        abnormal_indices = [i for i, c in enumerate(res['codes']) if c != 'N']
        
        for idx in abnormal_indices:
            if idx < len(res['peaks']):
                peak_loc = res['peaks'][idx]
                
                # Define a window around the peak (e.g., ±40 samples)
                start_p = max(0, int(peak_loc - 40))
                end_p = min(len(full_signal), int(peak_loc + 40))
                
                danger_signal[start_p:end_p] = full_signal[start_p:end_p]
                normal_signal[start_p:end_p] = np.nan # Xóa ở bên Normal để không bị trùng màu

        # Create DataFrame for plotting
        st.session_state.monitor_data = pd.DataFrame({
            'Normal': normal_signal,
            'Danger': danger_signal
        })
        
        st.session_state.peak_map = {p: c for p, c in zip(res['peaks'], res['codes'])}

    # 2. Start / Stop
    col_m1, col_m2 = st.columns([1, 5])
    with col_m1:
        # Toggle button logic
        if 'monitor_running' not in st.session_state:
            st.session_state.monitor_running = False

        if not st.session_state.monitor_running:
            if st.button("▶️ CHẠY MONITOR", type="primary", use_container_width=True):
                st.session_state.monitor_running = True
                st.rerun()
        else:
            if st.button("⏹️ DỪNG LẠI", type="secondary", use_container_width=True):
                st.session_state.monitor_running = False
                st.rerun()

    # 3. Real-time Monitor Logic
    monitor_placeholder = st.empty()
    
    WINDOW_SIZE = 600   # number of samples to show in one frame
    STEP = 15           # number of samples to move forward each iteration
    SPEED = 0.05        # delay in seconds between frames

    if st.session_state.monitor_running:
        data = st.session_state.monitor_data
        peak_map = st.session_state.peak_map
        total_len = len(data)
        
        # loop
        # use placeholder to update chart
        curr_idx = 0
        
        while st.session_state.monitor_running:
            end_idx = curr_idx + WINDOW_SIZE
            
            if end_idx < total_len:
                chunk = data.iloc[curr_idx:end_idx]
                slice_start = curr_idx
                slice_end = end_idx
            else:
                curr_idx = 0
                continue

            # Logic for status and BPM
            current_status = "🟢 NORMAL SINUS RHYTHM"
            status_color = "monitor-normal"
            bpm_display = "--"
            
            # Scan for peaks in current window
            peaks_in_window = [p for p in peak_map.keys() if slice_start <= p < slice_end]
            
            if peaks_in_window:
                last_peak = peaks_in_window[-1] # Get the last peak in the window
                code = peak_map[last_peak]
                bpm = np.random.randint(60, 90) if code == 'N' else np.random.randint(100, 160)
                bpm_display = f"{bpm}"
                
                if code != 'N':
                    info = CLASS_INFO[code]
                    current_status = f"⚠️ WARNING: {info['name']}"
                    status_color = "monitor-alarm"

            # Render UI in loop
            with monitor_placeholder.container():
                # Hàng thông số
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                    <div class="monitor-box {status_color}" style="font-size: 20px; padding: 15px;">
                        {current_status}
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="monitor-box" style="border-color: #00d4ff; color: #00d4ff;">
                        ❤️ {bpm_display} <span style="font-size:14px">BPM</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Line Chart
                st.line_chart(
                    chunk, 
                    color=["#00FF00", "#FF0000"], # Xanh lá cho Normal, Đỏ cho Danger
                    height=350,
                    use_container_width=True
                )

            curr_idx += STEP
            time.sleep(SPEED) 
    else:
        st.info("Nhấn 'CHẠY MONITOR' để bắt đầu phiên theo dõi.")