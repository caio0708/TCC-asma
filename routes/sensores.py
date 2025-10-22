# sensores.py

from flask import Blueprint, render_template, jsonify, current_app
import paho.mqtt.client as mqtt
import json
import threading
import os
from dotenv import load_dotenv
import joblib
from datetime import datetime, date 
import sqlite3
import time
from routes.api import get_weather, get_air_quality, get_user_location

sensores_bp = Blueprint('sensores', __name__)

# Carregar as variáveis do arquivo .env
load_dotenv()

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
MODELOS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
DB_PATH = os.path.join(DADOS_DIR, 'sensores.db')
MODEL_FILENAME = os.path.join(MODELOS_DIR, 'modelo_tosse_aprimorado.pkl')
SCALER_FILENAME = os.path.join(MODELOS_DIR, 'scaler_tosse_aprimorado.pkl')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# MQTT e atualizar sensores
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_ALL = "sensorestcc/#"

def get_today_cough_count_from_db():
    """Consulta o banco de dados para obter a contagem máxima de tosse para o dia atual."""
    conn = None
    try:
        today_str = date.today().strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Usar MAX() pois a contagem é cumulativa durante o dia, refletindo a lógica de insights.py
        cursor.execute('SELECT MAX("contagem-tosse") FROM sensores WHERE Data = ?', (today_str,))
        result = cursor.fetchone()
        # Se não houver registros para hoje, o resultado será (None,)
        if result and result[0] is not None:
            return int(result[0])
        return 0
    except Exception as e:
        print(f"Erro ao buscar contagem de tosse do dia no DB: {e}")
        return 0 # Retorna 0 em caso de erro para não quebrar a aplicação
    finally:
        if conn:
            conn.close()

# Colunas e Sensores Padrão 
DB_COLUMNS = [
    'Data', 'Hora', 'frequencia-respiratoria', 'batimentos-cardiacos', 'saturacao',
    'temperatura-corporal', 'temperatura-ambiente', 'temperatura-oximetro', 'qualidade-ar-pm25',
    'qualidade-ar-pm10', 'qualidade-ar-aqi', 'qualidade-ar-o3', 'qualidade-ar-no2', 'qualidade-ar-so2', 
    'piezo', 'contagem-tosse', 'som', 'umidade', 'acelerometro-x', 'acelerometro-y', 'acelerometro-z', 
    'giroscopio-x', 'giroscopio-y', 'giroscopio-z'
]

SENSORES_PADRAO = [
    {"id": "frequencia-respiratoria", "valor": 0, "unidade": "rpm"},
    {"id": "batimentos-cardiacos", "valor": 0, "unidade": "bpm"},
    {"id": "saturacao", "valor": 0, "unidade": "%"},
    {"id": "temperatura-corporal", "valor": 0, "unidade": "°C"},
    {"id": "temperatura-ambiente", "valor": 0, "unidade": "°C"},
    {"id": "temperatura-oximetro", "valor": 0, "unidade": "°C"},
    {"id": "qualidade-ar-pm25", "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-pm10", "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-aqi", "valor": 0, "unidade": ""},
    {"id": "qualidade-ar-o3", "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-no2", "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-so2", "valor": 0, "unidade": "µg/m³"},
    {"id": "piezo", "valor": 0, "unidade": "Hz"},
    {"id": "contagem-tosse", "valor": 0, "unidade": "no dia"},
    {"id": "som", "valor": 0, "unidade": "no dia"},
    {"id": "umidade", "valor": 0, "unidade": "%"},
    {"id": "acelerometro-x", "valor": 0, "unidade": "m/s²"},
    {"id": "acelerometro-y", "valor": 0, "unidade": "m/s²"},
    {"id": "acelerometro-z", "valor": 0, "unidade": "m/s²"},
    {"id": "giroscopio-x", "valor": 0, "unidade": "°/s"},
    {"id": "giroscopio-y", "valor": 0, "unidade": "°/s"},
    {"id": "giroscopio-z", "valor": 0, "unidade": "°/s"},
]
UNIDADES = {s['id']: s['unidade'] for s in SENSORES_PADRAO}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    columns_str = ', '.join([f'`{col}` REAL' if col not in ['Data', 'Hora'] else f'`{col}` TEXT' for col in DB_COLUMNS])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS sensores ({columns_str})")
    conn.commit()
    conn.close()

init_db()

def atualizar_dados_api_externa(app_state, state_lock):
    """
    Busca dados de APIs externas (localização, clima, ar) e atualiza 
    o estado central da aplicação periodicamente.
    """
    # Espera um pouco no início para a aplicação principal carregar
    time.sleep(5) 
    
    while True:
        try:
            print("API | Buscando dados de localização, clima e qualidade do ar...")
            # 1. Obter localização e chave da API
            lat, lon, city = get_user_location()
            api_key = os.getenv('API_WEATHER_KEY')

            # 2. Buscar dados de clima e qualidade do ar
            temp_api, humidity_api = get_weather(lat, lon)
            air_quality_data = get_air_quality(lat, lon, api_key)

            # 3. Atualizar o estado central de forma segura
            with state_lock:
                # Atualiza clima
                if temp_api is not None:
                    app_state['temperatura-ambiente'] = temp_api
                if humidity_api is not None:
                    app_state['umidade'] = humidity_api

                # Atualiza qualidade do ar se os dados foram recebidos com sucesso
                if air_quality_data and air_quality_data[0] is not None:
                    aqi, pm2_5, pm10, o3, no2, so2 = air_quality_data
                    app_state['qualidade-ar-aqi'] = aqi
                    app_state['qualidade-ar-pm25'] = pm2_5
                    app_state['qualidade-ar-pm10'] = pm10
                    app_state['qualidade-ar-o3'] = o3
                    app_state['qualidade-ar-no2'] = no2
                    app_state['qualidade-ar-so2'] = so2
            
            print(f"API | Estado atualizado com sucesso para a cidade: {city}.")

        except Exception as e:
            print(f"API | Erro no loop de atualização de dados externos: {e}")

        # Espera 15 minutos (900 segundos) antes da próxima atualização
        time.sleep(200)

# ===== FUNÇÃO DE SALVAR NO DB =====
def salvar_dados_db(app_state, state_lock):
    """
    Salva os dados do estado central da aplicação no banco de dados periodicamente.
    """
    time.sleep(10)
    while True:
        try:
            with state_lock:
                current_state = app_state.copy()

            # <<< LÓGICA DE TOSSE CORRIGIDA >>>
            # O app_state['contagem-tosse'] já é a fonte da verdade, inicializado com o valor do DB.

            # <<< ALTERADO: Monta a lista de valores a partir do estado central >>>
            valores_para_salvar = [
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S")
            ]
            for col in DB_COLUMNS[2:]:
                # Usa o valor do estado ou 0 se não existir
                valor = current_state.get(col, 0)
                # Garante que é um número
                try:
                    valores_para_salvar.append(float(valor))
                except (ValueError, TypeError):
                    valores_para_salvar.append(0)

            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.cursor()
            
            placeholders = ', '.join(['?'] * len(DB_COLUMNS))
            colunas_str = ', '.join([f'"{col}"' for col in DB_COLUMNS])
            sql = f"INSERT INTO sensores ({colunas_str}) VALUES ({placeholders})"
            
            cursor.execute(sql, valores_para_salvar)
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Erro no loop de salvar DB: {e}")
        
        time.sleep(30)

@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

@sensores_bp.route('/api/sensores')
def api_sensores():
    """
    <<< REESCRITO: Lê o estado central e o formata para a API. >>>
    """
    # Acessa o estado e o lock injetados pelo app principal
    app_state = current_app.config['app_state']
    state_lock = current_app.config['state_lock']

    data_formatada = []
    
    with state_lock:
        # Cria uma cópia segura do estado para trabalhar
        current_state = app_state.copy()

    # Itera sobre o template de sensores para garantir que todos sejam exibidos
    for sensor_template in SENSORES_PADRAO:
        sensor_id = sensor_template['id']
        
        # Pega o valor do estado central; se não existir, usa 0
        valor_atual = current_state.get(sensor_id, 0)

        # Formatação dos valores (igual à sua lógica anterior)
        if isinstance(valor_atual, (int, float)):
            if sensor_id == 'batimentos-cardiacos':
                valor_formatado = round(valor_atual)
            elif sensor_id in ['saturacao', 'temperatura-corporal', 'temperatura-ambiente', 'temperatura-oximetro', 'frequencia-respiratoria']:
                valor_formatado = round(valor_atual, 1)
            else:
                valor_formatado = valor_atual
        else:
            valor_formatado = valor_atual

        data_formatada.append({
            "id": sensor_id,
            "valor": valor_formatado,
            "unidade": sensor_template['unidade']
        })
    return jsonify(data_formatada)