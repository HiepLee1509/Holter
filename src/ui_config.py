import streamlit as st

def setup_page_config():
    """Cấu hình trang cơ bản"""
    st.set_page_config(
        layout="wide",
        page_title="AI Heart Guard | ECG Analysis",
        page_icon="🫀",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Chứa toàn bộ CSS tùy chỉnh"""
    st.markdown("""
    <style>
    /* Tổng thể */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Metrics Box */
    div[data-testid="metric-container"] {
        background-color: #1a1c24;
        border: 1px solid #30333d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Custom Card cho lời khuyên */
    .advice-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid;
        background-color: #1e2130;
    }
    .safe { border-color: #2ecc71; color: #e8f5e9; }
    .warning { border-color: #f1c40f; color: #fff9c4; }
    .danger { border-color: #e74c3c; color: #ffebee; }
    
    /* Highlight text */
    .highlight {
        font-weight: bold;
        color: #00d4ff;
    }
    
            /* THÊM CSS MỚI CHO MONITOR */
    .monitor-box {
        border: 2px solid #30333d;
        background-color: #000000;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-family: 'Courier New', monospace;
        color: #00d4ff;
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    .monitor-normal {
        border-color: #2ecc71;
        color: #2ecc71;
        box-shadow: 0 0 10px rgba(46, 204, 113, 0.2);
    }
    
    @keyframes blink-animation {
      0% { background-color: #3b0000; border-color: red; color: red; }
      50% { background-color: #800000; border-color: darkred; color: white; }
      100% { background-color: #3b0000; border-color: red; color: red; }
    }
    
    .monitor-alarm {
        animation: blink-animation 0.5s infinite;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)