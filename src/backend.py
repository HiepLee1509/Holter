import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks # Phát hiện đỉnh R
import pywt # Thư viện Wavelet
# --- CẤU HÌNH DỮ LIỆU & LỜI KHUYÊN ---

# Định nghĩa thông tin chi tiết cho 5 lớp (Classes)
CLASS_INFO = {
    'N': {
        "name": "Bình thường (Normal)",
        "color": "green",
        "advice": "Nhịp tim của bạn đang ở trạng thái ổn định. Hãy duy trì lối sống lành mạnh, tập thể dục đều đặn và ăn uống cân bằng."
    },
    'S': {
        "name": "Ngoại tâm thu trên thất (SVEB)",
        "color": "orange",
        "advice": "Thường lành tính nhưng có thể do căng thẳng, caffeine hoặc thiếu ngủ. Nên hạn chế chất kích thích, nghỉ ngơi hợp lý. Nếu thấy hồi hộp nhiều, hãy đi khám."
    },
    'V': {
        "name": "Ngoại tâm thu thất (VEB)",
        "color": "red",
        "advice": "Có thể gây cảm giác hẫng nhịp. Nguyên nhân có thể do rối loạn điện giải, bệnh tim nền hoặc stress. Cần theo dõi tần suất, nếu xuất hiện dày đặc hoặc gây chóng mặt, cần gặp bác sĩ tim mạch ngay."
    },
    'F': {
        "name": "Nhịp hỗn hợp (Fusion Beat)",
        "color": "purple",
        "advice": "Là sự kết hợp giữa nhịp bình thường và nhịp bất thường. Đây là dấu hiệu cần được bác sĩ chuyên khoa đánh giá kỹ hơn qua Holter ECG."
    },
    'Q': {
        "name": "Nhịp không xác định (Unknown)",
        "color": "gray",
        "advice": "Tín hiệu bị nhiễu hoặc không rõ ràng. Vui lòng kiểm tra lại thiết bị đo, tiếp xúc điện cực và đo lại trong trạng thái tĩnh. Hoặc đi khám chuyên khoa để được đánh giá chính xác hơn."
    }
}

CLASSES_KEYS = ['N', 'S', 'V', 'F', 'Q']

def load_arrhythmia_model(model_path="model\\ecg_model_code 17_t5.h5"):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return None

def get_model_input_length(model):
    """Tự động lấy độ dài input đầu vào của model"""
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
            return int(input_shape[1])      #input_shape = (None, 187, 1) = (batch_size, input_length, features)
        # Nếu không lấy được từ input_shape, thử lấy từ lớp đầu tiên
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_shape'):
            cfg_shape = first_layer.input_shape
            if cfg_shape and len(cfg_shape) >= 2 and cfg_shape[1] is not None:
                 return int(cfg_shape[1])
    except:
        pass
    return 187 # Fallback

def denoise_signal_wavelet(signal, wavelet='sym8', level=1):
    """Lọc nhiễu Wavelet"""
    if len(signal) < 10:
        return signal
    try:
        coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level) 
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745 
        thresh = sigma * np.sqrt(2 * np.log(len(signal)))
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, thresh, mode='soft'))
        denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')
        
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        elif len(denoised_signal) < len(signal):
            pad_width = len(signal) - len(denoised_signal)
            denoised_signal = np.pad(denoised_signal, (0, pad_width), 'edge')
        return denoised_signal
    except:
        return signal

def detect_and_segment(denoised_ecg_signal, r_peak_height=0.5, r_peak_distance=150, output_length=187):
    """Phát hiện đỉnh R và phân đoạn"""
    peaks, _ = find_peaks(denoised_ecg_signal, height=r_peak_height, distance=r_peak_distance)
    
    ratio_before = 99 / 187
    window_before = int(output_length * ratio_before)
    window_after = output_length - window_before - 1
    
    segments = []
    valid_peak_locations = []
    
    for peak_loc in peaks:
        start = peak_loc - window_before
        end = peak_loc + window_after + 1
        if start < 0 or end > len(denoised_ecg_signal):
            continue
        segment = denoised_ecg_signal[start : end]
        if len(segment) == output_length:
            segments.append(segment)
            valid_peak_locations.append(peak_loc)
        
    if not segments:
        return np.array([]), np.array([])
        
    return np.array(segments), np.array(valid_peak_locations)

def predict_from_segments(segments_array, model):
    """Dự đoán và trả về mã lớp (N, S, V...)"""
    if segments_array.ndim == 2:
        X = segments_array.reshape(-1, segments_array.shape[1], 1)  # Thêm chiều features=1
    else:
        X = segments_array

    y_pred_probs = model.predict(X)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Trả về mã ký tự (N, S, V...) để frontend tra cứu trong CLASS_INFO
    predicted_codes = [CLASSES_KEYS[i] for i in y_pred_indices]
    return predicted_codes, y_pred_indices

def calculate_hrv_metrics(peaks, fs=360):
    """
    Tính toán các chỉ số biến thiên nhịp tim (HRV) cơ bản.
    Input:
        peaks: mảng chứa vị trí (index) các đỉnh R
        fs: tần số lấy mẫu
    Output:
        dict chứa các chỉ số và dữ liệu vẽ biểu đồ
    """
    if len(peaks) < 2:
        return None
    
    # 1. Tính khoảng cách RR (RR intervals) ra đơn vị mili-giây (ms)
    # np.diff(peaks) là khoảng cách giữa các đỉnh liên tiếp (tính bằng số mẫu)
    rr_intervals = np.diff(peaks)
    rr_ms = (rr_intervals / fs) * 1000
    
    # 2. Tính các chỉ số HRV (Time-domain)
    # SDNN: Độ lệch chuẩn của các khoảng RR (Đánh giá sức khỏe tổng quát)
    sdnn = np.std(rr_ms)
    
    # RMSSD: Căn bậc hai của trung bình bình phương sự khác biệt giữa các khoảng RR liên tiếp
    # (Đánh giá hoạt động của hệ thần kinh phó giao cảm)
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    # Nhịp tim trung bình (BPM)
    mean_rr = np.mean(rr_ms)
    mean_bpm = 60000 / mean_rr
    
    # 3. Chuẩn bị dữ liệu Poincaré Plot
    # Trục X: RR[n], Trục Y: RR[n+1]
    poincare_x = rr_ms[:-1]
    poincare_y = rr_ms[1:]
    
    return {
        "rr_ms": rr_ms,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "mean_bpm": mean_bpm,
        "poincare_x": poincare_x,
        "poincare_y": poincare_y
    }

def analyze_batch_data(patient_data_map, model, fs=360, wavelet='sym8', r_peak_height=0.5):
    """
    Chạy phân tích hàng loạt trên toàn bộ dataset.
    Trả về DataFrame tóm tắt để hiển thị bảng.
    """
    results = []
    
    # Lấy độ dài input cần thiết
    required_len = get_model_input_length(model)
    
    # Duyệt qua từng bệnh nhân/bản ghi
    # Sử dụng enumerate để trả về tiến trình nếu cần
    total_files = len(patient_data_map)
    
    for idx, (pid, raw_signal) in enumerate(patient_data_map.items()):
        try:
            # 1. Chuyển đổi sang numpy array
            signal = np.array(raw_signal)
            
            # 2. Xử lý tín hiệu
            denoised = denoise_signal_wavelet(signal, wavelet=wavelet)
            segments, peaks = detect_and_segment(denoised, r_peak_height, output_length=required_len)
            
            stats = {
                "ID": pid,
                "Total Beats": 0,
                "BPM (Avg)": 0,
                "Status": "Error",
                "Risk Level": "Unknown",
                "N": 0, "S": 0, "V": 0, "F": 0, "Q": 0
            }

            if len(segments) > 0:
                # 3. Dự đoán
                pred_codes, _ = predict_from_segments(segments, model)
                
                # 4. Thống kê
                counts = pd.Series(pred_codes).value_counts()
                total_beats = len(pred_codes)
                
                # Tính nhịp tim trung bình
                if len(peaks) > 1:
                    avg_diff = np.mean(np.diff(peaks))
                    bpm = int(60 / (avg_diff / fs))
                else:
                    bpm = 0
                
                # Cập nhật stats
                stats["Total Beats"] = total_beats
                stats["BPM (Avg)"] = bpm
                stats["Status"] = "Success"
                
                # Fill số lượng từng loại
                for code in ['N', 'S', 'V', 'F', 'Q']:
                    count = counts.get(code, 0)
                    stats[code] = count
                
                # Đánh giá mức độ nguy hiểm
                if stats['V'] > 0 or stats['F'] > 0:
                    stats['Risk Level'] = "High 🔴"
                elif stats['S'] > 0:
                    stats['Risk Level'] = "Medium 🟡"
                else:
                    stats['Risk Level'] = "Low 🟢"
            else:
                stats["Status"] = "No Peaks Found"
                
            results.append(stats)
            
        except Exception as e:
            results.append({"ID": pid, "Status": f"Error: {str(e)}", "Risk Level": "Error"})

    return pd.DataFrame(results)