from flask import Blueprint, render_template, jsonify, request, session
from datetime import datetime, timedelta
import pandas as pd
import random
import uuid
from routes.api import get_weather, get_air_quality

import os
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import subprocess

insights_bp = Blueprint('insights', __name__)

lat = -23.5505
lon = -46.6333
API_KEY = '7288a386509b40eb0513fd8500bd5d5d'

UPLOAD_FOLDER = 'uploads/audio'
ALLOWED_EXTENSIONS = {'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_webm_to_wav(webm_path, wav_path):
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', webm_path, wav_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg error:\n{result.stderr.decode()}')

        print(f'✅ Conversão concluída: {wav_path}')
        return True
    except Exception as e:
        print(f'❌ Erro na conversão: {e}')
        return False

# Simulação de dados de coordenadas e API removida, usamos apenas random para ambiente
def get_ml_data():
    epochs = list(range(1, 11))
    accuracy = [round(random.uniform(0.7, 1.0), 2) for _ in epochs]
    loss = [round(random.uniform(0.1, 0.5), 2) for _ in epochs]
    return {"epochs": epochs, "accuracy": accuracy, "loss": loss}


def get_env_data():
    # 1) Leitura do CSV com tipos e colunas necessárias
    df = pd.read_csv(
        'dados/sensores.csv',
        dtype={'Data': str, 'Hora': str},
        usecols=[
            'Data', 'Hora', 'frequencia-respiratoria', 'batimentos-cardiacos',
            'saturacao', 'temperatura', 'qualidade-ar-pm25', 'qualidade-ar-pm10',
            'qualidade-ar-aqi', 'movimento-toracico', 'contagem-tosse'
        ]
    )

    # 2) Combina Data e Hora em datetime, ordena e limita aos 15 registros mais recentes
    df['timestamp'] = pd.to_datetime(
        df['Data'] + ' ' + df['Hora'],
        format='%Y-%m-%d %H:%M:%S', errors='coerce'
    )
    df.dropna(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    # 3) Agrupa timestamps até o mesmo minuto e remove duplicatas, mantendo o último de cada grupo
    df['timestamp_minute'] = df['timestamp'].dt.floor('min')  # truncate to minute
    df = df.drop_duplicates(subset=['timestamp_minute'], keep='last')

    # 4) Limita aos últimos 15 registros únicos
    df = df.tail(15)

    # 3) Formata o eixo X para data e horário mais amigável: dd/mm HH:MM
    labels = df['timestamp'].dt.strftime('%d/%m %H:%M').tolist()

    # 4) Extrai séries numéricas para a resposta JSON
    series = {
        'frequencia-respiratoria': df['frequencia-respiratoria'].tolist(),
        'batimentos-cardiacos': df['batimentos-cardiacos'].tolist(),
        'saturacao': df['saturacao'].tolist(),
        'temperatura': df['temperatura'].tolist(),
        'qualidade-ar-pm25': df['qualidade-ar-pm25'].tolist(),
        'qualidade-ar-pm10': df['qualidade-ar-pm10'].tolist(),
        'qualidade-ar-aqi': df['qualidade-ar-aqi'].tolist(),
        'movimento-toracico': df['movimento-toracico'].tolist(),
        'contagem-tosse': df['contagem-tosse'].tolist()
    }

    # 5) Dados externos (clima e qualidade do ar)
    temp, humidity = get_weather(lat, lon)
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY)

    # 6) Gatilhos iniciais baseados no primeiro ponto disponível
    triggers = []
    if series['qualidade-ar-pm25'] and series['qualidade-ar-pm25'][0] > 35:
        triggers.append('PM2.5 Alta')
    if series['qualidade-ar-pm10'] and series['qualidade-ar-pm10'][0] > 50:
        triggers.append('PM10 Alta')
    if series['qualidade-ar-aqi'] and series['qualidade-ar-aqi'][0] > 100:
        triggers.append('AQI Alto')
    if series['contagem-tosse'] and series['contagem-tosse'][0] > 5:
        triggers.append('Tosse Excessiva')
    if series['saturacao'] and series['saturacao'][0] < 92:
        triggers.append('Baixa Saturação')
    if series['frequencia-respiratoria'] and series['frequencia-respiratoria'][0] > 25:
        triggers.append('Respiração Acelerada')
    if temp and temp > 35:
        triggers.append('Calor Excessivo')

    return {
        'labels': labels,
        **series,
        'triggers': triggers
    }

# Rotas permanecem inalteradas
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@insights_bp.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Arquivo de áudio não enviado'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    webm_path = os.path.join(UPLOAD_FOLDER, filename)
    wav_filename = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)

    file.save(webm_path)

    if convert_webm_to_wav(webm_path, wav_path):
        return jsonify({
            'message': 'Áudio salvo e convertido com sucesso',
            'wav_path': f'/uploads/audio/{wav_filename}'  # ou caminho local completo
        }), 200
    else:
        return jsonify({'error': 'Falha na conversão do áudio'}), 500
