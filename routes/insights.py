# insights.py

from flask import Blueprint, render_template, jsonify, request, session
from datetime import datetime
import pandas as pd
import numpy as np
import io
import random
import uuid
import os
import subprocess

from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename
from routes.api import get_weather, get_air_quality

insights_bp = Blueprint('insights', __name__)

# --- CONFIGURAÇÕES ---
lat = -23.5505
lon = -46.6333
API_KEY = '7288a386509b40eb0513fd8500bd5d5d'
UPLOAD_FOLDER = 'Uploads/audio'
ALLOWED_EXTENSIONS = {'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CONSTANTES DE ANÁLISE ---
SAMPLING_RATE = 50  # Hz
MIN_SAMPLES = 2 * SAMPLING_RATE  # Minimum 2 seconds of data for analysis

# --- INÍCIO: NÚCLEO DE ANÁLISE ---

def analyze_ppg_robust(ir_signal, red_signal, fs):
    """Calcula HR, RR e SpO2 a partir de sinais PPG."""
    if len(ir_signal) < MIN_SAMPLES or len(red_signal) < MIN_SAMPLES:
        return 0, 0, 0, [], None, [], None, None

    try:
        # Filtro passa-banda para remover ruído e linha de base
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
    """Calcula a magnitude da aceleração e o nível de atividade."""
    if len(acc_x) < MIN_SAMPLES or len(acc_y) < MIN_SAMPLES or len(acc_z) < MIN_SAMPLES:
        return np.array([]), 0

    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    activity_signal = magnitude - np.mean(magnitude)
    activity_level = np.std(activity_signal) if len(activity_signal) > 0 else 0
    return magnitude, activity_level

def find_motion_peaks(motion_magnitude, fs):
    """Detecta picos de movimento (solavancos)."""
    if len(motion_magnitude) < MIN_SAMPLES:
        return [], 0
    motion_threshold = np.mean(motion_magnitude) + 3 * np.std(motion_magnitude)
    peaks, _ = find_peaks(motion_magnitude, height=motion_threshold, distance=fs * 0.3)
    return peaks, motion_threshold

def analyze_sound(mic_signal, fs):
    """Detecta eventos sonoros (picos)."""
    if len(mic_signal) < MIN_SAMPLES:
        return [], 0
    threshold = np.mean(mic_signal) + 2 * np.std(mic_signal)
    sound_peaks, _ = find_peaks(mic_signal, height=threshold, distance=fs * 0.3)
    return sound_peaks, threshold

def run_full_analysis(df_raw):
    """
    Executa a pipeline completa de análise de dados brutos dos sensores.
    Retorna um dicionário com os resultados e dados para os gráficos.
    """
    if len(df_raw) < MIN_SAMPLES:
        return {
            'results': {
                'heart_rate': 0, 'respiratory_rate': 0, 'spo2': 0, 'body_temp': 0,
                'activity_level': 0, 'sound_events': 0, 'motion_events': 0, 'cough_count': 0
            },
            'charts': {
                'time_axis': [],
                'ppg': {'signal': [], 'peaks_time': [], 'peaks_value': []},
                'respiration': {'time_axis': [], 'signal': [], 'peaks_time': [], 'peaks_value': []},
                'sound': {'signal': [], 'threshold': 0, 'peaks_time': [], 'peaks_value': []},
                'motion': {'signal': [], 'threshold': 0, 'peaks_time': [], 'peaks_value': []}
            }
        }

    # Extrai dados do DataFrame
    ir_data = df_raw['batimentos-cardiacos'].to_numpy()
    red_data = df_raw['saturacao'].to_numpy()
    mic_data = df_raw['contagem-tosse'].to_numpy()
    acc_x_data = df_raw['acelerometro-x'].to_numpy()
    acc_y_data = df_raw['acelerometro-y'].to_numpy()
    acc_z_data = df_raw['acelerometro-z'].to_numpy()
    fs = SAMPLING_RATE
    time_axis = (np.arange(len(ir_data)) / fs).tolist()

    # Executa as análises
    hr, rr, spo2, hr_peaks, ir_filtered, resp_signal, resp_peaks, resp_time_vector = analyze_ppg_robust(ir_data, red_data, fs)
    sound_peaks, mic_threshold = analyze_sound(mic_data, fs)
    motion_magnitude, activity = analyze_motion(acc_x_data, acc_y_data, acc_z_data)
    motion_peaks, motion_threshold = find_motion_peaks(motion_magnitude, fs)
    avg_body_temp = df_raw['tempBody'].mean() if 'tempBody' in df_raw else 0

    # Lógica de correlação para tosse
    cough_count = 0
    if len(sound_peaks) > 0 and len(motion_peaks) > 0:
        sound_times = sound_peaks / fs
        motion_times = motion_peaks / fs
        for st in sound_times:
            if np.any(np.abs(motion_times - st) < 0.1):
                cough_count += 1

    # Monta o dicionário de resultados
    results = {
        'heart_rate': round(float(hr), 1),
        'respiratory_rate': round(float(rr), 1),
        'spo2': round(float(spo2), 1),
        'body_temp': round(float(avg_body_temp), 2) if not np.isnan(avg_body_temp) else 0,
        'activity_level': round(float(activity), 3),
        'sound_events': len(sound_peaks),
        'motion_events': len(motion_peaks),
        'cough_count': cough_count
    }

    # Monta o dicionário com dados para os gráficos de forma segura
    ppg_chart_data = {'signal': [], 'peaks_time': [], 'peaks_value': []}
    if ir_filtered is not None and len(ir_filtered) > 0:
        ppg_chart_data['signal'] = ir_filtered.tolist()
        if len(hr_peaks) > 0:
            valid_peaks = hr_peaks[hr_peaks < len(ir_filtered)]
            ppg_chart_data['peaks_time'] = (valid_peaks / fs).tolist()
            ppg_chart_data['peaks_value'] = ir_filtered[valid_peaks].tolist()

    resp_chart_data = {'time_axis': [], 'signal': [], 'peaks_time': [], 'peaks_value': []}
    if resp_signal is not None and resp_time_vector is not None and len(resp_signal) > 0:
        resp_chart_data['time_axis'] = resp_time_vector.tolist()
        resp_chart_data['signal'] = resp_signal.tolist()
        if len(resp_peaks) > 0:
            valid_resp_peaks = resp_peaks[resp_peaks < len(resp_signal)]
            resp_chart_data['peaks_time'] = [resp_time_vector[i] for i in valid_resp_peaks if i < len(resp_time_vector)]
            resp_chart_data['peaks_value'] = [resp_signal[i] for i in valid_resp_peaks if i < len(resp_signal)]

    charts = {
        'time_axis': time_axis,
        'ppg': ppg_chart_data,
        'respiration': resp_chart_data,
        'sound': {
            'signal': mic_data.tolist() if len(mic_data) > 0 else [],
            'threshold': float(mic_threshold) if mic_threshold is not None else 0,
            'peaks_time': (sound_peaks / fs).tolist() if len(sound_peaks) > 0 else [],
            'peaks_value': [float(mic_data[i]) for i in sound_peaks if i < len(mic_data)]
        },
        'motion': {
            'signal': motion_magnitude.tolist() if len(motion_magnitude) > 0 else [],
            'threshold': float(motion_threshold) if motion_threshold is not None else 0,
            'peaks_time': (motion_peaks / fs).tolist() if len(motion_peaks) > 0 else [],
            'peaks_value': [float(motion_magnitude[i]) for i in motion_peaks if i < len(motion_magnitude)]
        }
    }

    return {'results': results, 'charts': charts}

# --- FIM: NÚCLEO DE ANÁLISE ---

def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_webm_to_wav(webm_path, wav_path):
    """Converte arquivo WEBM para WAV usando ffmpeg."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', webm_path, wav_path],
            capture_output=True, text=True, check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Erro na conversão com ffmpeg: {e.stderr}')
        return False
    except FileNotFoundError:
        print("❌ Comando 'ffmpeg' não encontrado. Certifique-se de que está instalado e no PATH do sistema.")
        return False
    except Exception as e:
        print(f'❌ Erro inesperado na conversão: {e}')
        return False

def get_ml_data():
    """Simula dados de treinamento de ML."""
    epochs = list(range(1, 11))
    accuracy = [round(random.uniform(0.7, 1.0), 2) for _ in epochs]
    loss = [round(random.uniform(0.1, 0.5), 2) for _ in epochs]
    return {"epochs": epochs, "accuracy": accuracy, "loss": loss}

def get_env_data():
    """
    Lê DADOS BRUTOS, executa a análise completa, busca dados externos
    e retorna tudo em um formato consolidado para a API.
    """
    try:
        df_raw = pd.read_csv('dados/sensores.csv')
        df_to_analyze = df_raw.tail(10 * SAMPLING_RATE)
        if len(df_to_analyze) < MIN_SAMPLES:
            return {
                "error": f"Dados insuficientes para análise. Pelo menos {MIN_SAMPLES} amostras (2s) são necessárias.",
                "analysis": {
                    "results": {
                        "heart_rate": 0, "respiratory_rate": 0, "spo2": 0, "body_temp": 0,
                        "activity_level": 0, "sound_events": 0, "motion_events": 0, "cough_count": 0
                    },
                    "charts": {
                        "time_axis": [],
                        "ppg": {"signal": [], "peaks_time": [], "peaks_value": []},
                        "respiration": {"time_axis": [], "signal": [], "peaks_time": [], "peaks_value": []},
                        "sound": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []},
                        "motion": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []}
                    }
                },
                "external_data": {"temperature": None, "humidity": None, "aqi": None, "pm2_5": None, "pm10": None},
                "triggers": ["Dados insuficientes para análise"]
            }
    except FileNotFoundError:
        return {
            "error": "Arquivo 'dados/raw_sensor_data.csv' não encontrado.",
            "analysis": {
                "results": {
                    "heart_rate": 0, "respiratory_rate": 0, "spo2": 0, "body_temp": 0,
                    "activity_level": 0, "sound_events": 0, "motion_events": 0, "cough_count": 0
                },
                "charts": {
                    "time_axis": [],
                    "ppg": {"signal": [], "peaks_time": [], "peaks_value": []},
                    "respiration": {"time_axis": [], "signal": [], "peaks_time": [], "peaks_value": []},
                    "sound": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []},
                    "motion": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []}
                }
            },
            "external_data": {"temperature": None, "humidity": None, "aqi": None, "pm2_5": None, "pm10": None},
            "triggers": ["Arquivo de dados não encontrado"]
        }
    except Exception as e:
        return {
            "error": f"Erro ao ler os dados: {e}",
            "analysis": {
                "results": {
                    "heart_rate": 0, "respiratory_rate": 0, "spo2": 0, "body_temp": 0,
                    "activity_level": 0, "sound_events": 0, "motion_events": 0, "cough_count": 0
                },
                "charts": {
                    "time_axis": [],
                    "ppg": {"signal": [], "peaks_time": [], "peaks_value": []},
                    "respiration": {"time_axis": [], "signal": [], "peaks_time": [], "peaks_value": []},
                    "sound": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []},
                    "motion": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []}
                }
            },
            "external_data": {"temperature": None, "humidity": None, "aqi": None, "pm2_5": None, "pm10": None},
            "triggers": [f"Erro ao processar dados: {e}"]
        }

    # Executar a análise completa dos sinais
    analysis_data = run_full_analysis(df_to_analyze)
    res = analysis_data['results']

    # Buscar dados externos
    temp, humidity = get_weather(lat, lon)
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY)

    # Definir gatilhos (triggers) com base nos resultados da análise
    triggers = []
    if pm2_5 and pm2_5 > 35: triggers.append('PM2.5 Alta')
    if pm10 and pm10 > 50: triggers.append('PM10 Alta')
    if aqi and aqi > 100: triggers.append('AQI Alto')
    if res['cough_count'] > 3: triggers.append('Tosse Excessiva')
    if res['spo2'] > 0 and res['spo2'] < 95: triggers.append('Baixa Saturação')
    if res['respiratory_rate'] > 25: triggers.append('Respiração Acelerada')
    if temp and temp > 35: triggers.append('Calor Excessivo')

    return {
        'analysis': analysis_data,
        'external_data': {
            'temperature': temp,
            'humidity': humidity,
            'aqi': aqi,
            'pm2_5': pm2_5,
            'pm10': pm10
        },
        'triggers': triggers
    }

# Rotas Flask
@insights_bp.route('/insights')
def insights():
    ml_data = get_ml_data()
    env_data = get_env_data()
    usage_events = session.get('usage_events', [])
    return render_template('insights.html', ml_data=ml_data, env_data=env_data, usage_events=usage_events)

@insights_bp.route('/api/data')
def api_data():
    return jsonify({
        'ml_data': get_ml_data(),
        'env_data': get_env_data(),
        'usage_events': session.get('usage_events', [])
    })

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
        return jsonify({'error': 'Extensão de arquivo não permitida ou arquivo inválido'}), 400

    filename = secure_filename(f"{uuid.uuid4()}.webm")
    webm_path = os.path.join(UPLOAD_FOLDER, filename)
    wav_filename = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)

    try:
        file.save(webm_path)
    except Exception as e:
        return jsonify({'error': f'Falha ao salvar o arquivo: {e}'}), 500

    if convert_webm_to_wav(webm_path, wav_path):
        return jsonify({
            'message': 'Áudio salvo e convertido com sucesso',
            'wav_path': f'/{UPLOAD_FOLDER}/{wav_filename}'
        }), 200
    else:
        return jsonify({'error': 'Falha na conversão do áudio'}), 500