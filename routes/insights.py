# insights.py

from flask import Blueprint, render_template, jsonify, request, session, current_app 
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import numpy as np
import os
from dotenv import load_dotenv
import subprocess
import threading
from routes.sensores import SENSORES_PADRAO
import sqlite3
import tensorflow as tf
import joblib
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename
from routes.api import get_weather, get_air_quality
import uuid
import librosa

from routes.PERF import predict_perf 

insights_bp = Blueprint('insights', __name__)

# Configurações
lat = -23.5505
lon = -46.6333
# Carregar as variáveis do arquivo .env
load_dotenv()
UPLOAD_FOLDER = 'Uploads/audio'
ALLOWED_EXTENSIONS = {'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "dados", "sensores.db")

# Configurações de ML
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'audio_asthma_detection_model.keras')
ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder.joblib')

ml_model = None
ml_encoder = None
try:
    if os.path.exists(MODEL_PATH):
        ml_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Modelo de áudio '{os.path.basename(MODEL_PATH)}' carregado.")
    else:
        print(f"⚠️ AVISO: Arquivo do modelo de áudio não encontrado em {MODEL_PATH}")
    if os.path.exists(ENCODER_PATH):
        ml_encoder = joblib.load(ENCODER_PATH)
        print(f"✅ Encoder de áudio '{os.path.basename(ENCODER_PATH)}' carregado.")
    else:
        print(f"⚠️ AVISO: Arquivo do encoder de áudio não encontrado em {ENCODER_PATH}")
except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar os artefatos de ML: {e}")

# Constantes de análise
SAMPLING_RATE = 50
MIN_SAMPLES = 2 * SAMPLING_RATE
CHART_WINDOW_SECONDS = 5

def predict_audio_class(wav_path):
    if not ml_model or not ml_encoder:
        return {"error": "Modelo de ML ou encoder não está carregado no servidor."}
    if not os.path.exists(wav_path):
        return {"error": f"Arquivo de áudio não encontrado em: {wav_path}"}
    try:
        y_new, sr_new = librosa.load(wav_path, sr=None)
        if len(y_new) < 2048:
            return {"error": "Amostra de áudio muito curta para análise."}
        mfcc_new = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
        X_new = np.mean(mfcc_new, axis=1)
        X_new_processed = X_new.reshape(1, X_new.shape[0], 1)
        y_new_pred_proba = ml_model.predict(X_new_processed)
        confidence = np.max(y_new_pred_proba)
        y_new_pred_encoded = np.argmax(y_new_pred_proba, axis=1)
        predicted_class = ml_encoder.inverse_transform(y_new_pred_encoded)[0]
        return {
            "predicted_class": str(predicted_class),
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"❌ Erro durante a predição de áudio para '{wav_path}': {e}")
        return {"error": f"Ocorreu um erro inesperado durante a análise do áudio: {str(e)}"}

def analyze_ppg_robust(ir_signal, red_signal, fs):
    if len(ir_signal) < MIN_SAMPLES or len(red_signal) < MIN_SAMPLES:
        return 0, 0, 0, [], None, [], None, None
    try:
        b, a = butter(3, [0.5, 4.5], btype='bandpass', fs=fs)
        ir_filtered = filtfilt(b, a, ir_signal)
    except ValueError as e:
        print(f"Erro no filtro PPG: {e}")
        return 0, 0, 0, [], None, [], None, None
    prominence_val = np.std(ir_filtered) / 3 if np.std(ir_filtered) > 0 else 0.1
    hr_peaks, _ = find_peaks(ir_filtered, distance=fs * 0.4, prominence=prominence_val)
    hr = 0
    if len(hr_peaks) > 1:
        avg_peak_interval_s = np.mean(np.diff(hr_peaks)) / fs
        hr = 60 / avg_peak_interval_s
        if not (40 <= hr <= 160): hr = 0
    rr, respiratory_signal, resp_peaks, resp_time_vector = 0, None, [], None
    if len(hr_peaks) > 3:
        peak_amplitudes = ir_filtered[hr_peaks]
        peak_times = hr_peaks / fs
        try:
            interp_func = interp1d(peak_times, peak_amplitudes, kind='cubic', fill_value="extrapolate")
            resp_time_vector = np.arange(peak_times[0], peak_times[-1], 1/fs)
            respiratory_signal = interp_func(resp_time_vector)
            resp_peaks, _ = find_peaks(respiratory_signal, distance=fs * 2.0)
            if len(resp_peaks) > 1:
                avg_resp_interval_s = np.mean(np.diff(resp_peaks)) / fs
                rr = 60 / avg_resp_interval_s
                if not (6 <= rr <= 35): rr = 0
        except ValueError as e:
            print(f"Erro na interpolação respiratória: {e}")
    spo2_est = 0
    if hr > 0 and len(red_signal) == len(ir_signal):
        R_values = []
        for i in range(len(hr_peaks) - 1):
            start, end = hr_peaks[i], hr_peaks[i+1]
            if start < len(ir_signal) and end <= len(ir_signal):
                ir_window, red_window = ir_signal[start:end], red_signal[start:end]
                ir_ac, red_ac = np.ptp(ir_window), np.ptp(red_window)
                ir_dc, red_dc = np.mean(ir_window), np.mean(red_window)
                if ir_ac > 0 and ir_dc > 0 and red_dc > 0 and red_ac > 0:
                    R = (red_ac / red_dc) / (ir_ac / ir_dc)
                    if 0.4 < R < 2.0: R_values.append(R)
        if R_values:
            avg_R = np.median(R_values)
            spo2_est = 110 - 25 * avg_R
            if spo2_est > 100: spo2_est = 100.0
            if spo2_est < 70: spo2_est = 0
    return hr, rr, spo2_est, hr_peaks, ir_filtered, respiratory_signal, resp_peaks, resp_time_vector

def analyze_motion(acc_x, acc_y, acc_z):
    if len(acc_x) < MIN_SAMPLES or len(acc_y) < MIN_SAMPLES or len(acc_z) < MIN_SAMPLES:
        return np.array([]), 0
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    thoracic_movement = np.std(magnitude) if len(magnitude) > 0 else 0
    return magnitude, thoracic_movement

def analyze_rotation(gyro_x, gyro_y, gyro_z):
    if len(gyro_x) < MIN_SAMPLES or len(gyro_y) < MIN_SAMPLES or len(gyro_z) < MIN_SAMPLES:
        return np.array([])
    rotation_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    return rotation_magnitude

def find_motion_peaks(motion_magnitude, fs):
    if len(motion_magnitude) < MIN_SAMPLES:
        return [], 0
    motion_threshold = np.mean(motion_magnitude) + 3 * np.std(motion_magnitude)
    peaks, _ = find_peaks(motion_magnitude, height=motion_threshold, distance=fs * 0.3)
    return peaks, motion_threshold

def analyze_sound(mic_signal, fs):
    if len(mic_signal) < MIN_SAMPLES:
        return [], 0
    threshold = np.mean(mic_signal) + 2 * np.std(mic_signal)
    sound_peaks, _ = find_peaks(mic_signal, height=threshold, distance=fs * 0.3)
    return sound_peaks, threshold

def analyze_piezo(piezo_signal, fs):
    if len(piezo_signal) < MIN_SAMPLES:
        return [], 0
    threshold = np.mean(piezo_signal) + 2 * np.std(piezo_signal)
    piezo_peaks, _ = find_peaks(piezo_signal, height=threshold, distance=fs * 0.3)
    return piezo_peaks, threshold

def analyze_weekly_cough(df):
    try:
        if 'Data' not in df.columns or 'contagem-tosse' not in df.columns:
           # print("Insights.py | analyze_weekly_cough: Colunas 'Data' ou 'contagem-tosse' ausentes")
            return None
        df_copy = df.copy()
        df_copy['Data'] = pd.to_datetime(df_copy['Data'], errors='coerce')
        df_copy.dropna(subset=['Data'], inplace=True)
        today = pd.to_datetime(datetime.now().date())
        seven_days_ago = today - pd.Timedelta(days=6)
        df_last_7_days = df_copy[df_copy['Data'] >= seven_days_ago]
        if df_last_7_days.empty:
          #  print("Insights.py | analyze_weekly_cough: Nenhum dado nos últimos 7 dias")
            dias_semana_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            return {
                "labels": [d[:3] for d in dias_semana_pt],
                "data": [0] * 7
            }
        daily_max_coughs = df_last_7_days.groupby(df_last_7_days['Data'].dt.date)['contagem-tosse'].max()
        daily_max_coughs.index = pd.to_datetime(daily_max_coughs.index)
        day_map = {
            0: 'Segunda-feira',
            1: 'Terça-feira',
            2: 'Quarta-feira',
            3: 'Quinta-feira',
            4: 'Sexta-feira',
            5: 'Sábado',
            6: 'Domingo'
        }
        cough_by_weekday = {day.weekday(): count for day, count in daily_max_coughs.items()}
        final_counts = [cough_by_weekday.get(i, 0) for i in range(7)]
        labels = [day_map[i][:3] for i in range(7)]
        #print(f"Insights.py | analyze_weekly_cough: Dados calculados - labels: {labels}, counts: {final_counts}")
        return {
            "labels": labels,
            "data": final_counts
        }
    except Exception as e:
        print(f"Insights.py | analyze_weekly_cough: Erro ao analisar tosse semanal: {e}")
        return None

def run_full_analysis(df_raw):
    if len(df_raw) < MIN_SAMPLES:
        print(f"Insights.py | run_full_analysis: Dados insuficientes ({len(df_raw)} amostras, mínimo {MIN_SAMPLES})")
        return {
            'results': {sensor['id']: sensor['valor'] for sensor in SENSORES_PADRAO},
            'charts': {
                'time_axis': [],
                'ppg': {'signal': [], 'peaks': []},
                'respiration': {'signal': [], 'peaks': []},
                'sound': {'signal': [], 'threshold': 0, 'peaks': []},
                'motion': {'signal': [], 'rotation_signal': [], 'threshold': 0, 'peaks': []},
                'piezo': {'signal': [], 'threshold': 0, 'peaks': []}
            }
        }
    time_axis = df_raw['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').tolist()
    fs = SAMPLING_RATE
    sensor_data = {col: df_raw[col].to_numpy() for col in df_raw.columns if col not in ['Data', 'Hora', 'Timestamp']}
    hr, rr, spo2, hr_peaks, ir_filtered, resp_signal, resp_peaks, resp_time_vector = analyze_ppg_robust(
        sensor_data.get('batimentos-cardiacos', np.array([])), sensor_data.get('saturacao', np.array([])), fs
    )
    motion_magnitude, _ = analyze_motion(
        sensor_data.get('acelerometro-x', np.array([])), sensor_data.get('acelerometro-y', np.array([])), sensor_data.get('acelerometro-z', np.array([]))
    )
    motion_peaks, motion_threshold = find_motion_peaks(motion_magnitude, fs)
    rotation_magnitude = analyze_rotation(
        sensor_data.get('giroscopio-x', np.array([])), sensor_data.get('giroscopio-y', np.array([])), sensor_data.get('giroscopio-z', np.array([]))
    )
    sound_signal = sensor_data.get('som', np.array([]))
    sound_peaks, sound_threshold = analyze_sound(sound_signal, fs)
    piezo_peaks, piezo_threshold = analyze_piezo(sensor_data.get('piezo', np.array([])), fs)
    body_temp = df_raw['temperatura-corporal'].mean() if 'temperatura-corporal' in df_raw and not df_raw['temperatura-corporal'].isna().all() else 0
    oximeter_temp = df_raw['temperatura-oximetro'].mean() if 'temperatura-oximetro' in df_raw and not df_raw['temperatura-oximetro'].isna().all() else 0
    results = {
        'frequencia-respiratoria': round(float(rr), 1) if rr > 0 else 0,
        'batimentos-cardiacos': round(float(hr), 1) if hr > 0 else 0,
        'saturacao': round(float(spo2), 1) if spo2 > 0 else 0,
        'temperatura-corporal': round(float(body_temp), 2) if not np.isnan(body_temp) else 0,
        'temperatura-ambiente': 0,
        'temperatura-oximetro': round(float(oximeter_temp), 2) if not np.isnan(oximeter_temp) else 0,
        'qualidade-ar-pm25': 0,
        'qualidade-ar-pm10': 0,
        'qualidade-ar-aqi': 0,
        'piezo': round(float(np.mean(sensor_data.get('piezo', np.array([])))), 3) if len(sensor_data.get('piezo', np.array([]))) > 0 else 0,
        'contagem-tosse': round(float(np.max(sensor_data.get('contagem-tosse', np.array([0])))), 0) if len(sensor_data.get('contagem-tosse', np.array([]))) > 0 else 0,
        'som': round(float(np.mean(sensor_data.get('som', np.array([])))), 3) if len(sensor_data.get('som', np.array([]))) > 0 else 0,
        'umidade': 0,
        'acelerometro-x': round(float(np.mean(sensor_data.get('acelerometro-x', np.array([])))), 3) if len(sensor_data.get('acelerometro-x', np.array([]))) > 0 else 0,
        'acelerometro-y': round(float(np.mean(sensor_data.get('acelerometro-y', np.array([])))), 3) if len(sensor_data.get('acelerometro-y', np.array([]))) > 0 else 0,
        'acelerometro-z': round(float(np.mean(sensor_data.get('acelerometro-z', np.array([])))), 3) if len(sensor_data.get('acelerometro-z', np.array([]))) > 0 else 0,
        'giroscopio-x': round(float(np.mean(sensor_data.get('giroscopio-x', np.array([])))), 3) if len(sensor_data.get('giroscopio-x', np.array([]))) > 0 else 0,
        'giroscopio-y': round(float(np.mean(sensor_data.get('giroscopio-y', np.array([])))), 3) if len(sensor_data.get('giroscopio-y', np.array([]))) > 0 else 0,
        'giroscopio-z': round(float(np.mean(sensor_data.get('giroscopio-z', np.array([])))), 3) if len(sensor_data.get('giroscopio-z', np.array([]))) > 0 else 0
    }
    charts = {
        'ppg': {
            'signal': create_signal_chart_data(time_axis, ir_filtered),
            'peaks': create_peak_chart_data(time_axis, hr_peaks, ir_filtered)
        },
        'respiration': {
            'signal': create_signal_chart_data(resp_time_vector, resp_signal, is_timestamp=False),
            'peaks': create_peak_chart_data(resp_time_vector, resp_peaks, resp_signal, is_timestamp=False)
        },
        'sound': {
            'signal': create_signal_chart_data(time_axis, sound_signal),
            'threshold': float(sound_threshold),
            'peaks': create_peak_chart_data(time_axis, sound_peaks, sound_signal)
        },
        'motion': {
            'signal': create_signal_chart_data(time_axis, motion_magnitude),
            'rotation_signal': create_signal_chart_data(time_axis, rotation_magnitude),
            'threshold': float(motion_threshold),
            'peaks': create_peak_chart_data(time_axis, motion_peaks, motion_magnitude)
        },
        'piezo': {
            'signal': create_signal_chart_data(time_axis, sensor_data.get('piezo', np.array([]))),
            'threshold': float(piezo_threshold),
            'peaks': create_peak_chart_data(time_axis, piezo_peaks, sensor_data.get('piezo', np.array([])))
        }
    }
    return {'results': results, 'charts': charts}

def create_signal_chart_data(time_axis, signal_data, is_timestamp=True):
    if time_axis is None or signal_data is None or len(time_axis) == 0 or len(signal_data) == 0 or len(time_axis) != len(signal_data):
        # print(f"Insights.py | create_signal_chart_data: Dados inválidos (time_axis: {len(time_axis)}, signal_data: {len(signal_data)})")
        return []
    try:
        if is_timestamp:
            return [{'x': t, 'y': float(v)} for t, v in zip(time_axis, signal_data) if not np.isnan(v)]
        else:
            return [{'x': float(t), 'y': float(v)} for t, v in zip(time_axis, signal_data) if not np.isnan(v)]
    except Exception as e:
        print(f"Insights.py | create_signal_chart_data: Erro ao criar dados do gráfico: {e}")
        return []

def create_peak_chart_data(time_axis, peak_indices, signal_data, is_timestamp=True):
    if time_axis is None or signal_data is None or len(peak_indices) == 0 or len(time_axis) == 0 or len(signal_data) == 0:
        # print(f"Insights.py | create_peak_chart_data: Dados insuficientes (peaks: {len(peak_indices)})")
        return []
    try:
        valid_peaks = [p for p in peak_indices if p < len(signal_data) and p < len(time_axis)]
        if is_timestamp:
            return [{'x': time_axis[i], 'y': float(signal_data[i])} for i in valid_peaks]
        else:
            return [{'x': float(time_axis[i]), 'y': float(signal_data[i])} for i in valid_peaks]
    except Exception as e:
        print(f"Insights.py | create_peak_chart_data: Erro ao criar dados de picos: {e}")
        return []

def get_empty_analysis_data():
    return {
        "analysis": {
            "results": {sensor['id']: sensor['valor'] for sensor in SENSORES_PADRAO},
            "charts": {
                "time_axis": [],
                "ppg": {"signal": [], "peaks": []},
                "respiration": {"signal": [], "peaks": []},
                "sound": {"signal": [], "threshold": 0, "peaks": []},
                "motion": {"signal": [], "rotation_signal": [], "threshold": 0, "peaks": []},
                "piezo": {"signal": [], "threshold": 0, "peaks": []}
            }
        },
        "external_data": {"temperatura-ambiente": None, "umidade": None},
        "weekly_cough_data": {"labels": [], "data": []}
    }

def get_historical_chart_data(conn, hours=1):
    """Busca dados históricos para os gráficos de Sinais Vitais e Ambiente."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    query = """
        SELECT "Data", "Hora", "batimentos-cardiacos", "saturacao", "temperatura-ambiente", "umidade"
        FROM sensores
        WHERE "Data" || ' ' || "Hora" >= ?
    """
    df = pd.read_sql_query(query, conn, params=(start_time.strftime('%Y-%m-%d %H:%M:%S'),))

    if df.empty:
        return None, None

    df['Timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    df = df.set_index('Timestamp').sort_index()

    # --- CORREÇÃO ---
    # Seleciona apenas as colunas numéricas que queremos agregar.
    # Isso evita que o .mean() tente calcular a média das colunas 'Data' e 'Hora'.
    numeric_cols = ["batimentos-cardiacos", "saturacao", "temperatura-ambiente", "umidade"]

    # Agrega os dados a cada 2 minutos para um gráfico mais limpo
    df_agg = df[numeric_cols].resample('2min').mean().interpolate(method='linear')

    labels = df_agg.index.strftime('%H:%M').tolist()

    vitals_chart = {
        "labels": labels,
        "bpm": df_agg['batimentos-cardiacos'].round(1).where(pd.notna(df_agg['batimentos-cardiacos']), None).tolist(),
        "spo2": df_agg['saturacao'].round(1).where(pd.notna(df_agg['saturacao']), None).tolist()
    }

    env_chart = {
        "labels": labels,
        "temp": df_agg['temperatura-ambiente'].round(1).where(pd.notna(df_agg['temperatura-ambiente']), None).tolist(),
        "humidity": df_agg['umidade'].round(1).where(pd.notna(df_agg['umidade']), None).tolist()
    }

    return vitals_chart, env_chart


def get_env_data():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # <<< OTIMIZAÇÃO: Definir o limite para os gráficos >>>
    # O gráfico de análise processa os últimos 5 segundos de dados (5s * 50Hz = 250 amostras)
    chart_limit = CHART_WINDOW_SECONDS * SAMPLING_RATE 
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # <<< OTIMIZAÇÃO: Query 1 (Pequena, para os gráficos de análise) >>>
        # Busca apenas os 250 registros mais recentes para a análise de janela
        df_latest_raw = pd.read_sql_query(f"SELECT * FROM sensores ORDER BY Data DESC, Hora DESC LIMIT {chart_limit}", conn)

        # <<< OTIMIZAÇÃO: Query 2 (Maior, mas só 2 colunas, para o gráfico de tosse semanal) >>>
        today = datetime.now().date()
        seven_days_ago = today - timedelta(days=6)
        df_weekly_cough_raw = pd.read_sql_query(
            'SELECT "Data", "contagem-tosse" FROM sensores WHERE "Data" >= ?', 
            conn, 
            params=(seven_days_ago.strftime('%Y-%m-%d'),)
        )
        
        # <<< OTIMIZAÇÃO: Query 3 (para os novos gráficos de histórico) >>>
        vitals_history_chart, env_history_chart = get_historical_chart_data(conn, hours=1)

        conn.close()
        
        # <<< Lógica de erro atualizada para usar o novo DataFrame >>>
        if df_latest_raw.empty:
            print("Insights.py | get_env_data: Banco de dados vazio")
            empty_data = get_empty_analysis_data()
            empty_data['error'] = "O banco de dados está vazio."
            return empty_data
            
        # Processa dados dos gráficos
        df_latest_raw['Timestamp'] = pd.to_datetime(df_latest_raw['Data'] + ' ' + df_latest_raw['Hora'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df_latest_raw.dropna(subset=['Timestamp'], inplace=True)
        
        # <<< Processa dados da tosse com o DataFrame otimizado >>>
        weekly_cough_analysis = analyze_weekly_cough(df_weekly_cough_raw.copy()) 
        
        analysis_data = None 
        if len(df_latest_raw) < MIN_SAMPLES:
            analysis_data = get_empty_analysis_data()['analysis']
        else:
            # <<< Usa o DataFrame otimizado para a análise >>>
            df_to_analyze = df_latest_raw.iloc[::-1].reset_index(drop=True) 
            analysis_data = run_full_analysis(df_to_analyze)
        
        res = analysis_data['results']

        # ... (leitura do app_state e remoção das chamadas de API de clima permanecem iguais) ...

        app_state_global = current_app.config['app_state']
        state_lock_global = current_app.config['state_lock']

        with state_lock_global:
            current_live_data = app_state_global.copy()
        
        for sensor_id, value in current_live_data.items():
            if sensor_id in res:
                try:
                    res[sensor_id] = float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit() else value
                except (ValueError, TypeError):
                    res[sensor_id] = value

        # <<< Lógica de contagem de tosse diária atualizada >>>
        if not df_latest_raw.empty:
            today_dt = pd.to_datetime(datetime.now().date())
            # Filtra o df_latest_raw (últimos 250) para hoje
            df_today = df_latest_raw[pd.to_datetime(df_latest_raw['Data']).dt.date == today_dt.date()]
            if not df_today.empty:
                res['contagem-tosse'] = int(df_today['contagem-tosse'].max())
            
            # Fallback: Se os 250 últimos não forem de hoje, verifica o Df semanal
            elif not df_weekly_cough_raw.empty:
                df_weekly_cough_raw['Data'] = pd.to_datetime(df_weekly_cough_raw['Data'], errors='coerce')
                df_today_weekly = df_weekly_cough_raw[df_weekly_cough_raw['Data'].dt.date == today_dt.date()]
                if not df_today_weekly.empty:
                        res['contagem-tosse'] = int(df_today_weekly['contagem-tosse'].max())
        
                # --- Sanitização para predict_perf: impede float(None) / float("") ---
        def _to_float(x, default=0.0):
            if x is None:
                return default
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                s = x.strip().replace(',', '.')
                try:
                    return float(s)
                except Exception:
                    return default
            return default

        # Converte todos os valores do res para float seguro
        perf_input = {k: _to_float(v) for k, v in res.items()}

        try:
            pefr_prediction = predict_perf(perf_input)
        except Exception as e:
            # Não deixe quebrar o fluxo: registre e devolva um erro tratável ao front
            print(f"❌ ERRO inesperado em predict_perf: {e}")
            pefr_prediction = {"error": f"Falha ao calcular PERF: {e}"}


        # ... (triggers e o resto da função permanecem iguais) ...

        triggers = []
        ### CORREÇÃO: Removido o erro de digitação 'saturação' com 'ç'.
        saturacao_val = res.get('saturacao') 
        
        if saturacao_val is not None and 0 < saturacao_val < 95:
            triggers.append('Baixa Saturação')

        if res.get('contagem-tosse', 0) > 5: triggers.append('Tosse Excessiva')
        if res.get('frequencia-respiratoria', 0) > 25: triggers.append('Respiração Acelerada')
        if res.get('temperatura-corporal', 0) > 38: triggers.append('Febre Detectada')
        if pefr_prediction and pefr_prediction.get("zone") == "RISK":
            triggers.append('Risco de Crise (PERF Baixo)')

        return {
            'timestamp': timestamp,
            'analysis': analysis_data,
            'triggers': triggers,
            'pefr_prediction': pefr_prediction,
            'weekly_cough_data': weekly_cough_analysis if weekly_cough_analysis else {"labels": [], "data": []},
            # Adiciona os dados dos novos gráficos ao retorno
            'vitals_history_chart': vitals_history_chart,
            'env_history_chart': env_history_chart
        }
    except (sqlite3.Error, ValueError, FileNotFoundError) as e:
        print(f"Insights.py | get_env_data: ERRO no processamento do DB: {e}")
        empty_data = get_empty_analysis_data()
        empty_data['error'] = str(e)
        return empty_data
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_webm_to_wav(webm_path, wav_path):
    try:
        subprocess.run(['ffmpeg', '-y', '-i', webm_path, wav_path], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Erro na conversão com ffmpeg: {e.stderr}')
        return False
    except FileNotFoundError:
        print("❌ 'ffmpeg' não encontrado. Instale-o e adicione-o ao PATH do sistema.")
        return False

@insights_bp.route('/insights')
def insights():
    env_data = get_env_data()
    usage_events = session.get('usage_events', [])
    #  Obtenha o nome de usuário da sessão
    username = session.get('username', 'Visitante')
    return render_template('insights.html', env_data=env_data, usage_events=usage_events,nome_usuario=username)

@insights_bp.route('/api/data')
def api_data():
    data = {
        'env_data': get_env_data(),
        'usage_events': session.get('usage_events', []),
    }
   # print(f"Insights.py | api_data: Retornando dados: {json.dumps(data, indent=2)}")
    return jsonify(data)

@insights_bp.route('/api/events', methods=['GET', 'POST', 'DELETE'])
def api_events():
    events = session.get('usage_events', [])
    if request.method == 'POST':
        data = request.json
        new_event = {'id': str(uuid.uuid4()), 'title': 'Uso da Bombinha', 'start': data['date']}
        events.append(new_event)
        session['usage_events'] = events
        return jsonify(new_event), 201
    if request.method == 'DELETE':
        event_id = request.args.get('id')
        events = [e for e in events if e['id'] != event_id]
        session['usage_events'] = events
        return '', 204
    return jsonify(events)

@insights_bp.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Arquivo de áudio não enviado'}), 400
    file = request.files['audio']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Extensão de arquivo não permitida'}), 400
    filename = secure_filename(f"{uuid.uuid4()}.webm")
    webm_path = os.path.join(UPLOAD_FOLDER, filename)
    wav_filename = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)
    try:
        file.save(webm_path)
    except Exception as e:
        return jsonify({'error': f'Falha ao salvar o arquivo: {e}'}), 500
    if convert_webm_to_wav(webm_path, wav_path):
        web_wav_path = f'/{UPLOAD_FOLDER}/{wav_filename}'.replace(os.path.sep, '/')
        return jsonify({'message': 'Áudio salvo e convertido', 'wav_path': web_wav_path}), 200
    else:
        return jsonify({'error': 'Falha na conversão do áudio'}), 500

@insights_bp.route('/api/predict-audio', methods=['POST'])
def predict_audio_route():
    data = request.get_json()
    wav_path = data.get('wav_path')
    if not wav_path:
        return jsonify({'error': 'Caminho do arquivo WAV não fornecido'}), 400
    local_wav_path = wav_path.lstrip('/')
    prediction_result = predict_audio_class(local_wav_path)
    if "error" in prediction_result:
        return jsonify(prediction_result), 500
    return jsonify(prediction_result), 200

@insights_bp.route('/api/historical_data')
def api_historical_data():
    """
    Endpoint que retorna dados históricos agregados para um sensor.
    Ex: /api/historical_data?sensor=batimentos-cardiacos&hours=24
    """
    sensor_name = request.args.get('sensor')
    hours = request.args.get('hours', 1, type=int) ### ALTERADO: Padrão de 24h para 1h

    if not sensor_name:
        return jsonify({"error": "Parâmetro 'sensor' é obrigatório."}), 400
    
    # Lista de sensores permitidos para evitar injeção de SQL
    allowed_sensors = [
        'batimentos-cardiacos','saturacao','temperatura-corporal','som','piezo',
        'acelerometro-x','acelerometro-y','acelerometro-z',
        'giroscopio-x','giroscopio-y','giroscopio-z',
        'contagem-tosse','frequencia-respiratoria'
    ]

    if sensor_name not in allowed_sensors:
        return jsonify({"error": "Sensor não permitido ou inválido."}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Query SQL segura (usamos aspas duplas para o nome da coluna)
        query = f'SELECT "Data", "Hora", "{sensor_name}" FROM sensores WHERE "Data" || " " || "Hora" >= ?'
        
        df = pd.read_sql_query(query, conn, params=(start_time.strftime('%Y-%m-%d %H:%M:%S'),))
        conn.close()

        if df.empty:
            return jsonify({"points": []}) # Retorna lista vazia se não houver dados

        df['Timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        # Remove valores onde o sensor é 0, exceto para tosse
        if sensor_name != 'contagem-tosse':
            df = df[df[sensor_name] != 0]

        df = df.set_index('Timestamp')
        
        # Define a regra de agregação (downsampling)
        agg_rule = '5T' # 5 minutos para 24h
        if hours <= 1:
            agg_rule = '10S' # 10 segundos se for 1 hora
        elif hours <= 6:
            agg_rule = '1T' # 1 minuto se for 6 horas
        
        # Usar .mean() para sensores analógicos, .max() para tosse
        if sensor_name == 'contagem-tosse':
            df_agg = df[sensor_name].resample(agg_rule).max()
        else:
            df_agg = df[sensor_name].resample(agg_rule).mean()

        # Preenche com 'None' (null no JSON) para criar lacunas no gráfico
        df_agg = df_agg.where(pd.notna(df_agg), None)
        
        # Formatar para Chart.js {x, y}
        chart_data = [
            {'x': idx.isoformat(), 'y': val}
            for idx, val in df_agg.items()
        ]
        
        return jsonify({"points": chart_data, "hours": hours, "rule": agg_rule}), 200, {'Cache-Control': 'no-store'}

    except Exception as e:
        print(f"Insights.py | api_historical_data: ERRO: {e}")
        return jsonify({"error": f"Erro ao processar dados históricos: {str(e)}"}), 500

@insights_bp.route('/api/latest_analysis_point')
def api_latest_analysis_point():
    """
    Endpoint super leve que retorna o último valor de um sensor específico
    com um timestamp, para gráficos "live" fluidos.
    """
    sensor_name = request.args.get('sensor')
    if not sensor_name:
        return jsonify({"error": "Parâmetro 'sensor' é obrigatório."}), 400

    app_state = current_app.config['app_state']
    state_lock = current_app.config['state_lock']

    with state_lock:
        value = app_state.get(sensor_name, 0)

    data = {
        'x': datetime.now().isoformat(),
        'y': value
    }
    return jsonify(data)

@insights_bp.route('/api/mpu6050_latest')
def api_mpu6050_latest():
    """
    Endpoint leve que retorna apenas os dados mais recentes do acelerômetro.
    """
    app_state = current_app.config['app_state']
    state_lock = current_app.config['state_lock']

    with state_lock:
        data = {
            'timestamp': datetime.now().isoformat(), 
            'accel_x': app_state.get('acelerometro-x', 0),
            'accel_y': app_state.get('acelerometro-y', 0),
            'accel_z': app_state.get('acelerometro-z', 0),
        
            'state': app_state.get('mpu_state', 0) 
        }
    return jsonify(data)

@insights_bp.route('/api/motion_history')
def motion_history():
    """
    Histórico agregado (magnitude) do movimento:
    - acc_mag = sqrt( (avg(ax))^2 + (avg(ay))^2 + (avg(az))^2 )
    - gyro_mag = sqrt( (avg(gx))^2 + (avg(gy))^2 + (avg(gz))^2 )
    """
    try:
        minutes = int(request.args.get('minutes', 60))
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        if minutes <= 120:
            rule_seconds = 10
        elif minutes <= 360:
            rule_seconds = 30
        elif minutes <= 720:
            rule_seconds = 60
        else:
            rule_seconds = 300

        con = sqlite3.connect(DB_PATH)
        sql = """
        SELECT
          strftime('%Y-%m-%dT%H:%M:%SZ',
                   CAST(strftime('%s', "Data" || ' ' || "Hora") / ? AS INTEGER) * ?) AS ts_bin,
          AVG("acelerometro-x") AS ax,
          AVG("acelerometro-y") AS ay,
          AVG("acelerometro-z") AS az,
          AVG("giroscopio-x") AS gx,
          AVG("giroscopio-y") AS gy,
          AVG("giroscopio-z") AS gz
        FROM sensores
        WHERE ("Data" || ' ' || "Hora") >= ?
          AND ("Data" || ' ' || "Hora") <= ?
        GROUP BY ts_bin
        ORDER BY ts_bin ASC
        """
        params = [rule_seconds, rule_seconds,
                  start_time.strftime('%Y-%m-%d %H:%M:%S'),
                  end_time.strftime('%Y-%m-%d %H:%M:%S')]
        cur = con.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()

        if not rows:
            return jsonify({"points": []})

        out = []
        for r in rows:
            ts, ax, ay, az, gx, gy, gz = r
            # trata None como 0 para não quebrar a magnitude
            ax = 0.0 if ax is None else float(ax)
            ay = 0.0 if ay is None else float(ay)
            az = 0.0 if az is None else float(az)
            gx = 0.0 if gx is None else float(gx)
            gy = 0.0 if gy is None else float(gy)
            gz = 0.0 if gz is None else float(gz)
            acc_mag = (ax*ax + ay*ay + az*az) ** 0.5
            gyro_mag = (gx*gx + gy*gy + gz*gz) ** 0.5
            out.append({"t": ts, "acc": acc_mag, "gyro": gyro_mag})

        return jsonify({"points": out, "minutes": minutes, "rule_seconds": rule_seconds})
    except Exception as e:
        print(f"❌ Erro em /api/motion_history: {e}")
        return jsonify({"error": str(e)}), 500


@insights_bp.route('/api/mpu6050_history')
def mpu6050_history():
    try:
        minutes = int(request.args.get('minutes', 180))  # 3h por padrão
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        # Define o tamanho do "bin" de agregação (em segundos)
        # Isso é a "aproximação" que você pediu
        if minutes <= 120:   # 2h ou menos
            rule_seconds = 10  # Agrupa a cada 10 segundos
        elif minutes <= 360: # 6h ou menos
            rule_seconds = 30  # Agrupa a cada 30 segundos
        elif minutes <= 720: # 12h ou menos
            rule_seconds = 60  # Agrupa a cada 1 minuto
        else:                # Até 24h
            rule_seconds = 300 # Agrupa a cada 5 minutos

        con = sqlite3.connect(DB_PATH)
        
        # OTIMIZAÇÃO: Fazer a agregação (resampling) diretamente no SQL
        # Isso é muito mais rápido do que carregar milhões de linhas no pandas
        sql = """
        SELECT
          -- Arredonda o timestamp para o início do intervalo (bin) e formata como ISO
          strftime('%Y-%m-%dT%H:%M:%SZ', CAST(strftime('%s', "Data" || ' ' || "Hora") / ? AS INTEGER) * ?) AS ts_bin,
          AVG("acelerometro-x") AS ax,
          AVG("acelerometro-y") AS ay,
          AVG("acelerometro-z") AS az
        FROM sensores
        WHERE ("Data" || ' ' || "Hora") >= ?
          AND ("Data" || ' ' || "Hora") <= ?
        GROUP BY ts_bin  -- Agrupa pelos intervalos de tempo
        ORDER BY ts_bin ASC
        """
        
        params = [
            rule_seconds, 
            rule_seconds,
            start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        cursor = con.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        con.close()

        if not rows:
            return jsonify({"points": []})

        # Formata a saída para o Chart.js
        # O pandas não é mais necessário aqui
        out = [{"t": row[0],
                "ax": (None if row[1] is None else float(row[1])),
                "ay": (None if row[2] is None else float(row[2])), # CORREÇÃO: Índices corretos
                "az": (None if row[3] is None else float(row[3]))} # CORREÇÃO: Índices corretos
               for row in rows]
               
        return jsonify({"points": out, "minutes": minutes, "rule_seconds": rule_seconds})

    except Exception as e:
        print(f"❌ Erro em /api/mpu6050_history: {e}")
        return jsonify({"error": str(e)}), 500
    
@insights_bp.route('/api/piezo_latest')
def api_piezo_latest():
    """
    Endpoint leve que retorna apenas a leitura mais recente do piezo
    (para o gráfico 'live' fluido).
    """
    app_state = current_app.config['app_state']
    state_lock = current_app.config['state_lock']
    with state_lock:
        data = {
            'timestamp': datetime.now().isoformat(),
            'piezo': app_state.get('piezo', 0)  # valor instantâneo do sinal
        }
    return jsonify(data)


@insights_bp.route('/api/piezo_history')
def piezo_history():
    """
    Histórico agregado do piezo, com bins adaptativos por intervalo (igual ao MPU).
    """
    try:
        minutes = int(request.args.get('minutes', 60))  # 1h por padrão
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        # Mesmo esquema de binagem do MPU【:contentReference[oaicite:2]{index=2}】
        if minutes <= 120:
            rule_seconds = 10
        elif minutes <= 360:
            rule_seconds = 30
        elif minutes <= 720:
            rule_seconds = 60
        else:
            rule_seconds = 300

        con = sqlite3.connect(DB_PATH)
        sql = """
        SELECT
          strftime('%Y-%m-%dT%H:%M:%SZ',
                   CAST(strftime('%s', "Data" || ' ' || "Hora") / ? AS INTEGER) * ?) AS ts_bin,
          AVG("piezo") AS pz
        FROM sensores
        WHERE ("Data" || ' ' || "Hora") >= ?
          AND ("Data" || ' ' || "Hora") <= ?
        GROUP BY ts_bin
        ORDER BY ts_bin ASC
        """
        params = [rule_seconds, rule_seconds,
                  start_time.strftime('%Y-%m-%d %H:%M:%S'),
                  end_time.strftime('%Y-%m-%d %H:%M:%S')]
        cur = con.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()

        if not rows:
            return jsonify({"points": []})

        out = [{"t": row[0],
                "pz": (None if row[1] is None else float(row[1]))}
               for row in rows]
        return jsonify({"points": out, "minutes": minutes, "rule_seconds": rule_seconds})
    except Exception as e:
        print(f"❌ Erro em /api/piezo_history: {e}")
        return jsonify({"error": str(e)}), 500
