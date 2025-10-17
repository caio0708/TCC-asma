# insights.py

from flask import Blueprint, render_template, jsonify, request, session
from datetime import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import subprocess
import threading
import json
import paho.mqtt.client as mqtt
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

# Configuração MQTT
live_data_lock = threading.Lock()
live_sensor_data = {s['id']: s['valor'] for s in SENSORES_PADRAO}

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_ALL = "sensorestcc/+"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        #print("Insights.py | Conectado ao broker MQTT")
        client.subscribe(TOPIC_ALL)
    else:
        print(f"Insights.py | Falha na conexão, código: {rc}")

def on_message(client, userdata, msg):
    try:
        sensor_id = msg.topic.split('/')[-1]
        payload = msg.payload.decode('utf-8')
        try:
            dados = json.loads(payload)
            valor = dados['valor'] if isinstance(dados, dict) and 'valor' in dados else dados
        except json.JSONDecodeError:
            try:
                valor = float(payload) if '.' in payload else int(payload)
            except ValueError:
                valor = payload
        with live_data_lock:
            live_sensor_data[sensor_id] = float(valor) if isinstance(valor, (int, float, str)) and str(valor).replace('.', '', 1).isdigit() else valor
          #  print(f"Insights.py | Atualizado {sensor_id}: {live_sensor_data[sensor_id]}")
    except Exception as e:
        print(f"Insights.py | Erro ao processar mensagem MQTT: {e}")

def start_mqtt_listener():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Insights.py | Erro ao conectar ao broker MQTT: {e}")

mqtt_thread = threading.Thread(target=start_mqtt_listener, daemon=True)
mqtt_thread.start()

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

def get_env_data():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = sqlite3.connect(DB_PATH)
        df_raw = pd.read_sql_query("SELECT * FROM sensores ORDER BY Data DESC, Hora DESC LIMIT 5000", conn)
        conn.close()
        if df_raw.empty:
            print("Insights.py | get_env_data: Banco de dados vazio")
            empty_data = get_empty_analysis_data()
            empty_data['error'] = "O banco de dados está vazio."
            return empty_data
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Data'] + ' ' + df_raw['Hora'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df_raw.dropna(subset=['Timestamp'], inplace=True)
        print(f"Insights.py | get_env_data: Número de amostras brutas: {len(df_raw)}")
        weekly_cough_analysis = analyze_weekly_cough(df_raw.copy())
        df_latest = df_raw.head(CHART_WINDOW_SECONDS * SAMPLING_RATE)  # 5 segundos * 50 Hz = 250 amostras
        print(f"Insights.py | get_env_data: Número de amostras para análise: {len(df_latest)}")
        if len(df_latest) < MIN_SAMPLES:
            print(f"Insights.py | get_env_data: Dados insuficientes para análise ({len(df_latest)} amostras)")
            analysis_data = get_empty_analysis_data()['analysis']
        else:
            df_to_analyze = df_latest.iloc[::-1].reset_index(drop=True)
            analysis_data = run_full_analysis(df_to_analyze)
        res = analysis_data['results']
        with live_data_lock:
            current_live_data = live_sensor_data.copy()
            for sensor_id, value in current_live_data.items():
                if sensor_id in res:
                    try:
                        res[sensor_id] = float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit() else value
                    except (ValueError, TypeError):
                        print(f"Insights.py | get_env_data: Valor inválido para {sensor_id}: {value}")
                        res[sensor_id] = value
        temp_api, humidity_api = get_weather(lat, lon)
        api_key = os.getenv('API_WEATHER_KEY')
        # Chama a função corrigida e verifica o resultado
        air_quality_data = get_air_quality(lat, lon, api_key)
        if air_quality_data:
            aqi, pm2_5, pm10, o3, no2, so2 = air_quality_data
        else:
            # Define valores padrão caso a API tenha falhado
            aqi, pm2_5, pm10, o3, no2, so2 = (None, None, None, None, None, None)
        # Atualiza o dicionário 'res' com os dados da API (ou com os valores padrão)
        res['temperatura-ambiente'] = round(float(temp_api), 2) if temp_api is not None else res.get('temperatura-ambiente', 0)
        res['umidade'] = round(float(humidity_api), 2) if humidity_api is not None else res.get('umidade', 0)
        res['qualidade-ar-aqi'] = int(aqi) if aqi is not None else res.get('qualidade-ar-aqi', 0)
        res['qualidade-ar-pm25'] = float(pm2_5) if pm2_5 is not None else res.get('qualidade-ar-pm25', 0)
        res['qualidade-ar-pm10'] = float(pm10) if pm10 is not None else res.get('qualidade-ar-pm10', 0)

        # Garantir que contagem-tosse seja o valor máximo do dia atual
        if not df_raw.empty:
            today = pd.to_datetime(datetime.now().date())
            df_today = df_raw[pd.to_datetime(df_raw['Data']) == today]
            if not df_today.empty:
                res['contagem-tosse'] = int(df_today['contagem-tosse'].max())
        
        # --- CÁLCULO DO PERF ---
        # Chama a função de previsão, passando os dados atuais dos sensores
        pefr_prediction = predict_perf(res)

        triggers = []
        if res.get('contagem-tosse', 0) > 3: triggers.append('Tosse Excessiva')
        if 0 < res.get('saturacao', 100) < 95: triggers.append('Baixa Saturação')
        if res.get('frequencia-respiratoria', 0) > 25: triggers.append('Respiração Acelerada')
        if res.get('temperatura-ambiente', 0) > 35: triggers.append('Calor Excessivo')
        if res.get('temperatura-corporal', 0) > 38: triggers.append('Febre Detectada')

        # Adiciona gatilho com base na previsão do PERF
        if pefr_prediction and pefr_prediction.get("zone") == "RISK":
            triggers.append('Risco de Crise (PERF Baixo)')

        return {
            'timestamp': timestamp,
            'analysis': analysis_data,
            'triggers': triggers,
            'pefr_prediction': pefr_prediction,  # Adiciona os dados de PERF na resposta
            'weekly_cough_data': weekly_cough_analysis if weekly_cough_analysis else {"labels": [], "data": []}
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