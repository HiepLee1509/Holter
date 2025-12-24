import streamlit as st
import pandas as pd
import plotly.express as px
from src.backend import analyze_batch_data, CLASS_INFO

def render_batch_analysis(patient_data_map, model, fs, wavelet_type, r_peak_height):
    """Hàm hiển thị giao diện quét hàng loạt"""
    
    st.markdown("### 🔍 Tổng quan dữ liệu toàn hệ thống")
    st.caption("Chế độ quét nhanh qua tất cả các bản ghi.")
    
    if st.button("🚀 BẮT ĐẦU QUÉT TOÀN BỘ DATASET", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Đang xử lý hàng loạt... Vui lòng chờ.")
        
        # Gọi hàm backend
        batch_df = analyze_batch_data(patient_data_map, model, fs, wavelet_type, r_peak_height)
        
        progress_bar.progress(100)
        status_text.text("✅ Hoàn tất!")
        st.session_state.batch_df = batch_df

    # HIỂN THỊ KẾT QUẢ
    if 'batch_df' in st.session_state:
        df = st.session_state.batch_df
        
        # 1. Metrics Tổng quan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số ca", len(df))
        c2.metric("Nguy cơ CAO", len(df[df['Risk Level'].str.contains("High")]))
        c3.metric("Nguy cơ TB", len(df[df['Risk Level'].str.contains("Medium")]))
        c4.metric("Tổng số nhịp", df['Total Beats'].sum())
        
        st.divider()
        
        # 2. Biểu đồ
        col_chart, col_table = st.columns([1, 2])
        
        with col_chart:
            risk_counts = df['Risk Level'].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values, 
                names=risk_counts.index, 
                title="Phân loại mức độ rủi ro",
                color=risk_counts.index,
                color_discrete_map={
                    "High 🔴": "#e74c3c",
                    "Medium 🟡": "#f1c40f",
                    "Low 🟢": "#2ecc71",
                    "Error": "gray"
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Biểu đồ tổng số lượng các loại nhịp
            total_counts = df[['N', 'S', 'V', 'F', 'Q']].sum()
            fig_bar = px.bar(
                x=total_counts.index, y=total_counts.values,
                title="Tổng số lượng nhịp phát hiện (Toàn bộ)",
                labels={'x': 'Loại nhịp', 'y': 'Số lượng'},
                color=total_counts.index,
                color_discrete_map={k: CLASS_INFO[k]['color'] for k in CLASS_INFO}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            st.subheader("📋 Bảng chi tiết từng bệnh nhân")
            
            # Highlight các dòng nguy hiểm
            st.dataframe(
                df,
                use_container_width=True,
                height=500,
                column_config={
                    "Risk Level": st.column_config.TextColumn(
                        "Đánh giá rủi ro",
                        help="High: Có VEB/Fusion. Medium: Có SVEB. Low: Normal.",
                        width="medium"
                    ),
                    "V": st.column_config.ProgressColumn(
                        "VEB (Nguy hiểm)",
                        format="%d",
                        min_value=0,
                        max_value=int(df['V'].max()) if len(df)>0 else 100,
                    ),
                }
            )
            
            # Download Button
            csv_batch = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tải báo cáo (CSV)", csv_batch, "batch_report.csv", "text/csv")