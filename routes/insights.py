from flask import Blueprint, render_template, jsonify, request, session
from datetime import datetime
import pandas as pd
import numpy as np
import io
import random
import uuid
import os
import subprocess

# --- NEW: Imports from the analysis script ---
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
# --- End of New Imports ---

from werkzeug.utils import secure_filename
from routes.api import get_weather, get_air_quality

insights_bp = Blueprint('insights', __name__)

# --- CONFIGURAÇÕES ---
lat = -23.5505
lon = -46.6333
API_KEY = '7288a386509b40eb0513fd8500bd5d5d'
UPLOAD_FOLDER = 'uploads/audio'
ALLOWED_EXTENSIONS = {'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CONSTANTES DE ANÁLISE ---
SAMPLING_RATE = 50  # Hz

# --- INÍCIO: NÚCLEO DE ANÁLISE (Portado do seu script original) ---

def analyze_ppg_robust(ir_signal, red_signal, fs):
    """Calcula HR, RR e SpO2 a partir de sinais PPG."""
    try:
        # Filtro passa-banda para remover ruído e linha de base
        b, a = butter(3, [0.5, 4.5], btype='bandpass', fs=fs)
        ir_filtered = filtfilt(b, a, ir_signal)
    except ValueError:
        # Retorna zero se o sinal for muito curto para o filtro
        return 0, 0, 0, [], None, [], None, None

    prominence_val = np.std(ir_filtered) / 3
    hr_peaks, _ = find_peaks(ir_filtered, distance=fs * 0.4, prominence=prominence_val)

    hr = 0
    if len(hr_peaks) > 1:
        avg_peak_interval_s = np.mean(np.diff(hr_peaks)) / fs
        hr = 60 / avg_peak_interval_s
        if not (40 <= hr <= 160): hr = 0

    rr, respiratory_signal, resp_peaks, resp_time_vector = 0, None, [], None
    if len(hr_peaks) > 5:
        peak_amplitudes = ir_filtered[hr_peaks]
        peak_times = hr_peaks / fs
        interp_func = interp1d(peak_times, peak_amplitudes, kind='cubic', fill_value="extrapolate")
        resp_time_vector = np.arange(peak_times[0], peak_times[-1], 1/fs)
        respiratory_signal = interp_func(resp_time_vector)
        resp_peaks, _ = find_peaks(respiratory_signal, distance=fs * 2.0)

        if len(resp_peaks) > 1:
            avg_resp_interval_s = np.mean(np.diff(resp_peaks)) / fs
            rr = 60 / avg_resp_interval_s
            if not (6 <= rr <= 35): rr = 0

    spo2_est = 0
    if hr > 0:
        R_values = []
        for i in range(len(hr_peaks) - 1):
            start, end = hr_peaks[i], hr_peaks[i+1]
            ir_window, red_window = ir_signal[start:end], red_signal[start:end]
            ir_ac, red_ac = np.ptp(ir_window), np.ptp(red_window)
            ir_dc, red_dc = np.mean(ir_window), np.mean(red_window)

            if ir_ac > 0 and ir_dc > 0 and red_dc > 0:
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
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    activity_signal = magnitude - np.mean(magnitude)
    activity_level = np.std(activity_signal)
    return magnitude, activity_level

def find_motion_peaks(motion_magnitude, fs):
    """Detecta picos de movimento (solavancos)."""
    motion_threshold = np.mean(motion_magnitude) + 3 * np.std(motion_magnitude)
    peaks, _ = find_peaks(motion_magnitude, height=motion_threshold, distance=fs * 0.3)
    return peaks, motion_threshold

def analyze_sound(mic_signal, fs):
    """Detecta eventos sonoros (picos)."""
    threshold = np.mean(mic_signal) + 2 * np.std(mic_signal)
    sound_peaks, _ = find_peaks(mic_signal, height=threshold, distance=fs * 0.3)
    return sound_peaks, threshold

def run_full_analysis(df_raw):
    """
    Executa a pipeline completa de análise de dados brutos dos sensores.
    Retorna um dicionário com os resultados e dados para os gráficos.
    """
    # Extrai dados do DataFrame
    ir_data = df_raw['ir'].to_numpy()
    red_data = df_raw['red'].to_numpy()
    mic_data = df_raw['mic'].to_numpy()
    acc_x_data = df_raw['accX'].to_numpy()
    acc_y_data = df_raw['accY'].to_numpy()
    acc_z_data = df_raw['accZ'].to_numpy()
    fs = SAMPLING_RATE
    time_axis = (np.arange(len(ir_data)) / fs).tolist()

    # Executa as análises
    hr, rr, spo2, hr_peaks, ir_filtered, resp_signal, resp_peaks, resp_time_vector = analyze_ppg_robust(ir_data, red_data, fs)
    sound_peaks, mic_threshold = analyze_sound(mic_data, fs)
    motion_magnitude, activity = analyze_motion(acc_x_data, acc_y_data, acc_z_data)
    motion_peaks, motion_threshold = find_motion_peaks(motion_magnitude, fs)
    avg_body_temp = df_raw['tempBody'].mean()

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
        'heart_rate': round(hr, 1),
        'respiratory_rate': round(rr, 1),
        'spo2': round(spo2, 1),
        'body_temp': round(avg_body_temp, 2),
        'activity_level': round(activity, 3),
        'sound_events': len(sound_peaks),
        'motion_events': len(motion_peaks),
        'cough_count': cough_count
    }

    # Monta o dicionário com dados para os gráficos (JSON-friendly)
    charts = {
        'time_axis': time_axis,
        'ppg': {
            'signal': ir_filtered.tolist() if ir_filtered is not None else [],
            'peaks_time': (hr_peaks / fs).tolist(),
            'peaks_value': ir_filtered[hr_peaks].tolist() if ir_filtered is not None and len(hr_peaks) > 0 else []
        },
        'respiration': {
            'time_axis': resp_time_vector.tolist() if resp_time_vector is not None else [],
            'signal': resp_signal.tolist() if resp_signal is not None else [],
            'peaks_time': (resp_time_vector[resp_peaks]).tolist() if resp_time_vector is not None and resp_signal is not None and len(resp_peaks) > 0 else [],
            'peaks_value': resp_signal[resp_peaks].tolist() if resp_signal is not None and len(resp_peaks) > 0 else []
        },
        'sound': {
            'signal': mic_data.tolist(),
            'threshold': mic_threshold,
            'peaks_time': (sound_peaks / fs).tolist(),
            'peaks_value': mic_data[sound_peaks].tolist()
        },
        'motion': {
            'signal': motion_magnitude.tolist(),
            'threshold': motion_threshold,
            'peaks_time': (motion_peaks / fs).tolist(),
            'peaks_value': motion_magnitude[motion_peaks].tolist()
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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg error:\n{result.stderr.decode()}')
        return True
    except Exception as e:
        print(f'❌ Erro na conversão: {e}')
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
        # Passo 1: Ler os dados brutos.
        # ATENÇÃO: Crie um arquivo 'dados/raw_sensor_data.csv' com as colunas:
        # accX,accY,accZ,gyroX,gyroY,gyroZ,mic,ir,red,tempBody
        df_raw = pd.read_csv('dados/raw_sensor_data.csv')
        # Pega as últimas N amostras para análise (ex: 10 segundos de dados)
        df_to_analyze = df_raw.tail(10 * SAMPLING_RATE)
    except FileNotFoundError:
        # Retorna um erro claro se o arquivo de dados brutos não existir
        return {"error": "Arquivo 'dados/raw_sensor_data.csv' não encontrado."}
    except Exception as e:
        return {"error": f"Erro ao ler os dados: {e}"}

    # Passo 2: Executar a análise completa dos sinais
    analysis_data = run_full_analysis(df_to_analyze)
    res = analysis_data['results']

    # Passo 3: Buscar dados externos
    temp, humidity = get_weather(lat, lon)
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY)

    # Passo 4: Definir gatilhos (triggers) com base nos resultados da análise
    triggers = []
    if pm2_5 and pm2_5 > 35: triggers.append('PM2.5 Alta')
    if pm10 and pm10 > 50: triggers.append('PM10 Alta')
    if aqi and aqi > 100: triggers.append('AQI Alto')
    if res['cough_count'] > 3: triggers.append('Tosse Excessiva')
    if res['spo2'] > 0 and res['spo2'] < 95: triggers.append('Baixa Saturação')
    if res['respiratory_rate'] > 25: triggers.append('Respiração Acelerada')
    if temp and temp > 35: triggers.append('Calor Excessivo')

    # Passo 5: Consolidar todos os dados para a resposta da API
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
    # Nota: A renderização do template agora depende de uma estrutura de dados diferente.
    # O seu 'insights.html' precisará ser adaptado para ler de `env_data['analysis']['results']`
    # e `env_data['analysis']['charts']`.
    ml_data = get_ml_data()
    env_data = get_env_data()
    usage_events = session.get('usage_events', [])
    return render_template('insights.html', ml_data=ml_data, env_data=env_data, usage_events=usage_events)

@insights_bp.route('/api/data')
def api_data():
    # Esta rota agora retorna a análise completa e os dados externos
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
    if not allowed_file(file.filename):
        return jsonify({'error': 'Extensão de arquivo não permitida'}), 400

    filename = secure_filename(file.filename)
    webm_path = os.path.join(UPLOAD_FOLDER, filename)
    wav_filename = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)

    file.save(webm_path)

    if convert_webm_to_wav(webm_path, wav_path):
        return jsonify({
            'message': 'Áudio salvo e convertido com sucesso',
            'wav_path': f'/uploads/audio/{wav_filename}'
        }), 200
    else:
        return jsonify({'error': 'Falha na conversão do áudio'}), 500

