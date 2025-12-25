import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from src.backend import CLASS_INFO, CLASSES_KEYS

# MATPLOTLIB PLOTS
def set_plot_style(dark_mode=True):
    """Cấu hình style cho Matplotlib"""
    if dark_mode:
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": "#0e1117",
            "figure.facecolor": "#0e1117",
            "grid.color": "#444444",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white"
        })
    else:
        plt.style.use('default')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#e6e6e6",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black"
        })

def plot_raw_signal_with_peaks(raw_ecg, peaks, predicted_codes, dark_mode=False):
    set_plot_style(dark_mode)
    fig, ax = plt.subplots(figsize=(15, 5))
    
    line_color = "#00d4ff" if dark_mode else "#1f77b4"
    ax.plot(raw_ecg, label="Tín hiệu ECG", color=line_color, alpha=0.8, linewidth=1.2)
    
    # Draw peaks with different colors based on predicted classes
    for code in CLASSES_KEYS:
        info = CLASS_INFO[code]
        mask = np.array(predicted_codes) == code
        class_peaks = peaks[mask]
        
        if len(class_peaks) > 0:
            ax.scatter(class_peaks, raw_ecg[class_peaks], 
                       color=info['color'], 
                       label=info['name'], 
                       s=70, zorder=5, edgecolors='white' if dark_mode else 'black')

    ax.set_title("Phân loại trên toàn bộ tín hiệu")
    ax.set_xlabel("Mẫu (Sample)")
    ax.set_ylabel("Biên độ (mV)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_beat_segment(beat_data, pred_code=None, dark_mode=False):
    set_plot_style(dark_mode)
    fig, ax = plt.subplots(figsize=(8, 3))
    
    info = CLASS_INFO.get(pred_code, {'name': 'Unknown', 'color': 'gray'})
    
    ax.plot(beat_data.squeeze(), color=info['color'], linewidth=2)
    
    title = f"Nhịp: {info['name']}" if pred_code else "Hình dạng nhịp tim"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig

# PLOTLY INTERACTIVE PLOTS
def plot_interactive_ecg(raw_signal, peaks, codes, fs=360):
    """Vẽ biểu đồ ECG tương tác với Plotly"""
    # Create time axis in seconds instead of samples
    time_axis = np.arange(len(raw_signal)) / fs
    
    fig = go.Figure()
    
    # Draw raw ECG signal
    fig.add_trace(go.Scatter(
        x=time_axis, y=raw_signal,
        mode='lines',
        name='Tín hiệu ECG',
        line=dict(color='#00d4ff', width=1.5),
        opacity=0.8
    ))
    
    # Draw peaks with different colors based on predicted classes
    unique_codes = list(set(codes))
    for code in unique_codes:
        indices = [i for i, x in enumerate(codes) if x == code]
        # Get peak positions and amplitudes
        current_peaks = peaks[indices]
        current_peaks_time = current_peaks / fs
        current_amps = raw_signal[current_peaks]
        
        info = CLASS_INFO[code]
        
        fig.add_trace(go.Scatter(
            x=current_peaks_time, y=current_amps,
            mode='markers',
            name=f"{info['name']} ({code})",
            marker=dict(size=10, color=info['color'], line=dict(width=1, color='white')),
            hovertemplate="<b>%{text}</b><br>Time: %{x:.2f}s<br>Amp: %{y:.2f}<extra></extra>",
            text=[info['name']] * len(indices)
        ))

    fig.update_layout(
        title="Biểu đồ Điện tâm đồ",
        template="plotly_dark",
        xaxis_title="Thời gian (s)",
        yaxis_title="Biên độ (mV)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig 

def plot_classes_pie(codes):
    """Vẽ biểu đồ tròn tỷ lệ các loại nhịp (Pie Chart)"""
    if not codes or len(codes) == 0:
        return None
        
    # Get number of occurrences for each class
    counts = pd.Series(codes).value_counts()
    
    # Create DataFrame for plotting
    df_pie = pd.DataFrame({'Loại': counts.index, 'Số lượng': counts.values})
    df_pie['Tên'] = df_pie['Loại'].apply(lambda x: CLASS_INFO[x]['name'])
    
    # Plot pie chart
    fig_pie = px.pie(
        df_pie, 
        values='Số lượng', 
        names='Tên', 
        hole=0.4, 
        title="Tỷ lệ phân bố các loại nhịp"
    )
    fig_pie.update_traces(textinfo='percent+label')
    return fig_pie

def plot_beat_shape(segment, code, beat_index=0):
    """Vẽ hình thái chi tiết của một nhịp tim cụ thể (Line Chart)"""
    info = CLASS_INFO.get(code, {'name': 'Unknown', 'color': 'gray'})
    
    # squeeze() for 2D array with single column
    fig_beat = px.line(segment.squeeze(), title=f"Hình thái nhịp thứ {beat_index}")
    fig_beat.update_traces(line_color=info['color'], line_width=3)
    fig_beat.update_layout(
        xaxis_title="Mẫu (Sample)", 
        yaxis_title="Biên độ (mV)", 
        showlegend=False, 
        height=300
    )
    return fig_beat

def plot_poincare_chart(hrv_data):
    """Vẽ biểu đồ Poincaré từ dữ liệu HRV"""
    if not hrv_data: 
        return None
    
    fig_poincare = go.Figure()
    
    # Plot RR_n vs RR_n+1 points
    fig_poincare.add_trace(go.Scatter(
        x=hrv_data['poincare_x'],
        y=hrv_data['poincare_y'],
        mode='markers',
        name='Nhịp tim',
        marker=dict(
            size=8,
            color=hrv_data['poincare_y'], 
            colorscale='Viridis',
            showscale=True,
            line=dict(width=1, color='white')
        ),
        hovertemplate="RR(n): %{x:.1f}ms<br>RR(n+1): %{y:.1f}ms<extra></extra>"
    ))

    # Plot Identity Line
    min_val = min(np.min(hrv_data['poincare_x']), np.min(hrv_data['poincare_y']))
    max_val = max(np.max(hrv_data['poincare_x']), np.max(hrv_data['poincare_y']))
    fig_poincare.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Đường chuẩn',
        line=dict(color='white', width=1, dash='dash')
    ))

    fig_poincare.update_layout(
        title="Poincaré Plot (RR_n vs RR_n+1)",
        xaxis_title="RR_n (ms)",
        yaxis_title="RR_n+1 (ms)",
        template="plotly_dark",
        height=500,
        width=500,
        showlegend=False
    )
    return fig_poincare
