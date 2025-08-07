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

# --- INÍCIO: Adições para dados ao vivo ---
import threading
import json
import paho.mqtt.client as mqtt
from routes.sensores import SENSORES_PADRAO # Reutiliza a lista padrão
# --- FIM: Adições para dados ao vivo ---

# --- INÍCIO: Adições para ML ---
import librosa
import tensorflow as tf
import joblib
# --- FIM: Adições para ML ---

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

# --- INÍCIO: CONFIGURAÇÕES E CARREGAMENTO DO MODELO DE ML ---
MODEL_ARTIFACTS_DIR = r'E:\Dev\TCC-asma\ia\model_artifacts'
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'audio_asthma_detection_model.keras')
ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder.joblib')

ml_model = None
ml_encoder = None
try:
    if os.path.exists(MODEL_PATH):
        ml_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Modelo de áudio '{os.path.basename(MODEL_PATH)}' carregado.")
    else:
        print(f"⚠️  AVISO: Arquivo do modelo de áudio não encontrado em {MODEL_PATH}")

    if os.path.exists(ENCODER_PATH):
        ml_encoder = joblib.load(ENCODER_PATH)
        print(f"✅ Encoder de áudio '{os.path.basename(ENCODER_PATH)}' carregado.")
    else:
        print(f"⚠️  AVISO: Arquivo do encoder de áudio não encontrado em {ENCODER_PATH}")

except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar os artefatos de ML: {e}")
    ml_model = None
    ml_encoder = None
# --- FIM: CONFIGURAÇÕES E CARREGAMENTO DO MODELO DE ML ---


# --- CONSTANTES DE ANÁLISE ---
SAMPLING_RATE = 50  # Hz
MIN_SAMPLES = 2 * SAMPLING_RATE  # Mínimo de 2 segundos de dados para análise

# --- INÍCIO: LÓGICA DE DADOS AO VIVO (MQTT) ---

live_data_lock = threading.Lock()
# Inicializa o dicionário de dados ao vivo com a estrutura padrão
live_sensor_data = {s['id']: s['valor'] for s in SENSORES_PADRAO}

# Configuração MQTT
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_ALL = "sensorestcc/+"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Insights.py | Conectado ao broker MQTT")
        client.subscribe(TOPIC_ALL)
    else:
        print(f"Insights.py | Falha na conexão, código: {rc}")

def on_message(client, userdata, msg):
    """Callback para atualizar o dicionário de dados ao vivo."""
    try:
        sensor_id = msg.topic.split('/')[-1]
        payload = msg.payload.decode('utf-8')
        # Tenta decodificar como JSON, se não, trata como valor bruto
        try:
            dados = json.loads(payload)
            valor = dados['valor'] if isinstance(dados, dict) and 'valor' in dados else dados
        except json.JSONDecodeError:
            try:
                valor = float(payload) if '.' in payload else int(payload)
            except ValueError:
                valor = payload
        
        with live_data_lock:
            live_sensor_data[sensor_id] = valor
            # print(f"Insights.py | Atualizado {sensor_id}: {valor}") # Descomente para depuração

    except Exception as e:
        print(f"Insights.py | Erro ao processar mensagem MQTT: {e}")

def start_mqtt_listener():
    """Inicia o cliente MQTT em uma thread separada."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Insights.py | Erro ao conectar ao broker MQTT: {e}")

# Inicia a thread do MQTT
mqtt_thread = threading.Thread(target=start_mqtt_listener, daemon=True)
mqtt_thread.start()

# --- FIM: LÓGICA DE DADOS AO VIVO (MQTT) ---

# --- INÍCIO: FUNÇÃO DE PREDIÇÃO DE ÁUDIO ---
def predict_audio_class(wav_path):
    """
    Analisa um arquivo de áudio WAV e prevê a classe usando o modelo de ML treinado.
    """
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

        predicted_class_array = ml_encoder.inverse_transform(y_new_pred_encoded)
        predicted_class = predicted_class_array[0]

        return {
            "predicted_class": str(predicted_class),
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"❌ Erro durante a predição de áudio para '{wav_path}': {e}")
        return {"error": f"Ocorreu um erro inesperado durante a análise do áudio: {str(e)}"}
# --- FIM: FUNÇÃO DE PREDIÇÃO DE ÁUDIO ---

# --- INÍCIO: NÚCLEO DE ANÁLISE (INALERADO) ---

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
    """Calcula a magnitude da aceleração e o nível de movimento torácico."""
    if len(acc_x) < MIN_SAMPLES or len(acc_y) < MIN_SAMPLES or len(acc_z) < MIN_SAMPLES:
        return np.array([]), 0

    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    thoracic_movement = np.std(magnitude) if len(magnitude) > 0 else 0
    return magnitude, thoracic_movement

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
            'results': {sensor['id']: sensor['valor'] for sensor in SENSORES_PADRAO},
            'charts': {
                'time_axis': [],
                'ppg': {'signal': [], 'peaks_time': [], 'peaks_value': []},
                'respiration': {'time_axis': [], 'signal': [], 'peaks_time': [], 'peaks_value': []},
                'sound': {'signal': [], 'threshold': 0, 'peaks_time': [], 'peaks_value': []},
                'motion': {'signal': [], 'threshold': 0, 'peaks_time': [], 'peaks_value': []}
            }
        }

    # Extrai dados do DataFrame
    sensor_data = {sensor['id']: df_raw[sensor['id']].to_numpy() if sensor['id'] in df_raw else np.array([]) for sensor in SENSORES_PADRAO}
    fs = SAMPLING_RATE
    time_axis = (np.arange(len(df_raw)) / fs).tolist()

    # Executa as análises
    hr, rr, spo2, hr_peaks, ir_filtered, resp_signal, resp_peaks, resp_time_vector = analyze_ppg_robust(
        sensor_data['batimentos-cardiacos'], sensor_data['saturacao'], fs
    )
    sound_peaks, mic_threshold = analyze_sound(sensor_data['som'], fs)
    motion_magnitude, thoracic_movement = analyze_motion(
        sensor_data['acelerometro-x'], sensor_data['acelerometro-y'], sensor_data['acelerometro-z']
    )
    motion_peaks, motion_threshold = find_motion_peaks(motion_magnitude, fs)
    
    # Médias para sensores escalares
    body_temp = df_raw['temperatura-corporal'].mean() if 'temperatura-corporal' in df_raw and not df_raw['temperatura-corporal'].isna().all() else 0
    oximeter_temp = df_raw['temperatura-oximetro'].mean() if 'temperatura-oximetro' in df_raw and not df_raw['temperatura-oximetro'].isna().all() else 0

    # Monta o dicionário de resultados
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
        'movimento-toracico': round(float(thoracic_movement), 3) if thoracic_movement > 0 else 0,
        'contagem-tosse': len(sound_peaks) if sound_peaks is not None else 0,
        'som': round(float(np.mean(sensor_data['som'])), 3) if len(sensor_data['som']) > 0 else 0,
        'umidade': 0,
        'acelerometro-x': round(float(np.mean(sensor_data['acelerometro-x'])), 3) if len(sensor_data['acelerometro-x']) > 0 else 0,
        'acelerometro-y': round(float(np.mean(sensor_data['acelerometro-y'])), 3) if len(sensor_data['acelerometro-y']) > 0 else 0,
        'acelerometro-z': round(float(np.mean(sensor_data['acelerometro-z'])), 3) if len(sensor_data['acelerometro-z']) > 0 else 0,
        'giroscopio-x': round(float(np.mean(sensor_data['giroscopio-x'])), 3) if len(sensor_data['giroscopio-x']) > 0 else 0,
        'giroscopio-y': round(float(np.mean(sensor_data['giroscopio-y'])), 3) if len(sensor_data['giroscopio-y']) > 0 else 0,
        'giroscopio-z': round(float(np.mean(sensor_data['giroscopio-z'])), 3) if len(sensor_data['giroscopio-z']) > 0 else 0
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
            'signal': sensor_data['som'].tolist() if len(sensor_data['som']) > 0 else [],
            'threshold': float(mic_threshold) if mic_threshold is not None else 0,
            'peaks_time': (sound_peaks / fs).tolist() if len(sound_peaks) > 0 else [],
            'peaks_value': [float(sensor_data['som'][i]) for i in sound_peaks if i < len(sensor_data['som'])]
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

def get_empty_analysis_data():
    """Retorna uma estrutura de dados de análise vazia em caso de erro."""
    return {
        "analysis": {
            "results": {sensor['id']: sensor['valor'] for sensor in SENSORES_PADRAO},
            "charts": {
                "time_axis": [],
                "ppg": {"signal": [], "peaks_time": [], "peaks_value": []},
                "respiration": {"time_axis": [], "signal": [], "peaks_time": [], "peaks_value": []},
                "sound": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []},
                "motion": {"signal": [], "threshold": 0, "peaks_time": [], "peaks_value": []}
            }
        },
        "external_data": { "temperatura-ambiente": None, "umidade": None },
    }

def get_env_data():
    """
    Combina análise histórica do CSV (para gráficos) com dados ao vivo do MQTT (para cartões).
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        csv_path = r'E:\Dev\TCC-asma\ia\dados\sensores.csv' 
        # ATENÇÃO: Use o cabeçalho corrigido no seu CSV para esta parte funcionar
        df_raw = pd.read_csv(csv_path)
        df_raw.dropna(inplace=True)

        # Analisa os últimos 5 segundos para os gráficos
        df_to_analyze = df_raw.tail(5 * SAMPLING_RATE) 
        
        if len(df_to_analyze) < MIN_SAMPLES:
            analysis_data = get_empty_analysis_data()['analysis']
            triggers = ["Dados insuficientes no CSV para análise completa"]
        else:
            analysis_data = run_full_analysis(df_to_analyze)
            triggers = [] 

    except FileNotFoundError:
        return { "error": f"Arquivo '{csv_path}' não encontrado.", "timestamp": timestamp, **get_empty_analysis_data(), "triggers": ["Arquivo de dados CSV não encontrado"] }
    except Exception as e:
        # Adicionado mais detalhes ao erro para facilitar a depuração
        print(f"ERRO EM get_env_data: {e}")
        return { "error": f"Erro ao ler ou analisar o CSV: {e}", "timestamp": timestamp, **get_empty_analysis_data(), "triggers": [f"Erro no processamento do CSV: {e}"] }

    # 2. Mesclar com Dados ao Vivo (MQTT) para os cartões de resultado
    res = analysis_data['results']
    with live_data_lock:
        current_live_data = live_sensor_data.copy()
        # Sobrescreve os valores no dicionário 'results' com os dados mais recentes do MQTT
        for sensor_id, value in current_live_data.items():
            if sensor_id in res:
                # Mantém a formatação/tipo original se possível, mas atualiza o valor
                try:
                    res[sensor_id] = type(res[sensor_id])(value)
                except (ValueError, TypeError):
                    res[sensor_id] = value # Se a conversão falhar, apenas atribui

    # 3. Buscar Dados Externos (API de Clima/Ar)
    temp, humidity = get_weather(lat, lon)
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY)

    # Atualizar resultados com dados externos
    res['temperatura-ambiente'] = round(float(temp), 2) if temp is not None else res.get('temperatura-ambiente', 0)
    res['umidade'] = round(float(humidity), 2) if humidity is not None else res.get('umidade', 0)
    res['qualidade-ar-aqi'] = int(aqi) if aqi is not None else res.get('qualidade-ar-aqi', 0)
    res['qualidade-ar-pm25'] = float(pm2_5) if pm2_5 is not None else res.get('qualidade-ar-pm25', 0)
    res['qualidade-ar-pm10'] = float(pm10) if pm10 is not None else res.get('qualidade-ar-pm10', 0)

    # 4. Definir Gatilhos (triggers) com base nos dados mais recentes (já mesclados)
    triggers = []
    if res.get('contagem-tosse', 0) > 3: triggers.append('Tosse Excessiva')
    if 0 < res.get('saturacao', 100) < 95: triggers.append('Baixa Saturação')
    if res.get('frequencia-respiratoria', 0) > 25: triggers.append('Respiração Acelerada')
    if res.get('temperatura-ambiente', 0) > 35: triggers.append('Calor Excessivo')
    if res.get('temperatura-corporal', 0) > 38: triggers.append('Febre Detectada')

    return {
        'timestamp': timestamp,
        'analysis': analysis_data, # Contém 'results' atualizados e 'charts' do CSV
        'triggers': triggers
    }

# --- ROTAS FLASK e LÓGICA DE UPLOAD ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_webm_to_wav(webm_path, wav_path):
    try:
        # O argumento -y sobrescreve o arquivo de saída se ele já existir
        subprocess.run(['ffmpeg', '-y', '-i', webm_path, wav_path], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Erro na conversão com ffmpeg: {e.stderr}')
        return False
    except FileNotFoundError:
        print("❌ 'ffmpeg' não encontrado. Instale-o e adicione-o ao PATH do sistema.")
        return False
    return False

@insights_bp.route('/insights')
def insights():
    env_data = get_env_data()
    usage_events = session.get('usage_events', [])
    return render_template('insights.html', env_data=env_data, usage_events=usage_events)

@insights_bp.route('/api/data')
def api_data():
    return jsonify({
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
        # Retorna o caminho web, não o caminho do sistema de arquivos
        web_wav_path = f'/{UPLOAD_FOLDER}/{wav_filename}'.replace(os.path.sep, '/')
        return jsonify({'message': 'Áudio salvo e convertido', 'wav_path': web_wav_path}), 200
    else:
        return jsonify({'error': 'Falha na conversão do áudio'}), 500

# --- INÍCIO: NOVA ROTA PARA PREDIÇÃO DE ÁUDIO ---
@insights_bp.route('/api/predict-audio', methods=['POST'])
def predict_audio_route():
    data = request.get_json()
    wav_path = data.get('wav_path')

    if not wav_path:
        return jsonify({'error': 'Caminho do arquivo WAV não fornecido'}), 400

    # Converte o caminho web (ex: /Uploads/audio/file.wav) para um caminho de sistema de arquivos local
    # lstrip('/') remove a barra inicial para criar um caminho relativo correto
    local_wav_path = wav_path.lstrip('/')
    
    prediction_result = predict_audio_class(local_wav_path)
    
    if "error" in prediction_result:
        return jsonify(prediction_result), 500
        
    return jsonify(prediction_result), 200
# --- FIM: NOVA ROTA PARA PREDIÇÃO DE ÁUDIO ---