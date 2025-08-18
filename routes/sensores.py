# sensores.py

from flask import Blueprint, render_template, jsonify
import paho.mqtt.client as mqtt
import json
import threading
import os
import joblib
from datetime import datetime, date 
import sqlite3
import time
from routes.api import get_weather, get_air_quality, get_user_location
from routes.cough_detector import live_cough_counter

sensores_bp = Blueprint('sensores', __name__)

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
MODELOS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
DB_PATH = os.path.join(DADOS_DIR, 'sensores.db')
MODEL_FILENAME = os.path.join(MODELOS_DIR, 'modelo_tosse.pkl')
SCALER_FILENAME = os.path.join(MODELOS_DIR, 'scaler_tosse.pkl')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
sensor_lock = threading.Lock()

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

# Variável para controlar o reset diário
LAST_COUGH_RESET_DAY = date.today()
COUGH_COUNT = get_today_cough_count_from_db()

# Colunas e Sensores Padrão 
DB_COLUMNS = [
    'Data', 'Hora', 'frequencia-respiratoria', 'batimentos-cardiacos', 'saturacao',
    'temperatura-corporal', 'temperatura-ambiente', 'temperatura-oximetro', 'qualidade-ar-pm25',
    'qualidade-ar-pm10', 'qualidade-ar-aqi', 'piezo', 'contagem-tosse', 'som',
    'umidade', 'acelerometro-x', 'acelerometro-y', 'acelerometro-z', 'giroscopio-x',
    'giroscopio-y', 'giroscopio-z'
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
    {"id": "piezo", "valor": 0, "unidade": "Hz"},
    {"id": "contagem-tosse", "valor": 0, "unidade": "no dia"},
    {"id": "som", "valor": 0, "unidade": "no dia"},
    {"id": "umidade", "valor": 0, "unidade": "%"},
    {"id": "acelerometro-x", "valor": 0, "unidade": ""},
    {"id": "acelerometro-y", "valor": 0, "unidade": ""},
    {"id": "acelerometro-z", "valor": 0, "unidade": ""},
    {"id": "giroscopio-x", "valor": 0, "unidade": ""},
    {"id": "giroscopio-y", "valor": 0, "unidade": ""},
    {"id": "giroscopio-z", "valor": 0, "unidade": ""},
]
UNIDADES = {s['id']: s['unidade'] for s in SENSORES_PADRAO}

def inicializar_sensores():
    lat, lon, city = get_user_location()
    API_KEY = '7288a386509b40eb0513fd8500bd5d5d'
    temp, humidity = get_weather(lat, lon)
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY)
    sensores = [dict(s) for s in SENSORES_PADRAO]
    with sensor_lock:
        for s in sensores:
            if s['id'] == 'umidade':
                s['valor'] = float(humidity) if humidity is not None else 0
            elif s['id'] == 'qualidade-ar-pm25':
                s['valor'] = float(pm2_5) if pm2_5 is not None else 0
            elif s['id'] == 'qualidade-ar-pm10':
                s['valor'] = float(pm10) if pm10 is not None else 0
            elif s['id'] == 'qualidade-ar-aqi':
                s['valor'] = float(aqi) if aqi is not None else 0
            elif s['id'] == 'temperatura-ambiente':
                s['valor'] = float(temp) if temp is not None else 0
    return sensores

lista_sensores = inicializar_sensores()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    columns_str = ', '.join([f'`{col}` REAL' if col not in ['Data', 'Hora'] else f'`{col}` TEXT' for col in DB_COLUMNS])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS sensores ({columns_str})")
    conn.commit()
    conn.close()

init_db()

# MQTT e atualizar sensores
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_ALL = "sensorestcc/+"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado ao broker MQTT")
        client.subscribe(TOPIC_ALL)
    else:
        print(f"Falha na conexão, código: {rc}")

def atualizar_sensor(sensor_id, valor):
    with sensor_lock:
        for sensor in lista_sensores:
            if sensor['id'] == sensor_id:
                try:
                    sensor['valor'] = float(valor) if isinstance(valor, (int, float, str)) and str(valor).replace('.', '', 1).isdigit() else valor
                 #   print(f"--- Sensor '{sensor_id}' atualizado para: {sensor['valor']} ---")
                    return
                except ValueError:
                    print(f"--- Valor inválido para '{sensor_id}': {valor}, mantendo valor atual ---")
                    return
        unidade = UNIDADES.get(sensor_id, '')
        lista_sensores.append({'id': sensor_id, 'valor': float(valor) if isinstance(valor, (int, float, str)) and str(valor).replace('.', '', 1).isdigit() else valor, 'unidade': unidade})
        #print(f"--- Sensor novo '{sensor_id}' adicionado com valor: {valor} ---")

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
        atualizar_sensor(sensor_id, valor)
    except Exception as e:
        print(f"Erro ao processar mensagem MQTT: {e}")

def salvar_dados_db():
    global LAST_COUGH_RESET_DAY, COUGH_COUNT
    time.sleep(10)
    while True:
        try:
            today = date.today()
            with sensor_lock:
                # Verificar se é um novo dia para resetar o contador
                if today != LAST_COUGH_RESET_DAY:
                    COUGH_COUNT = 0
                    for sensor in lista_sensores:
                        if sensor['id'] == 'contagem-tosse':
                            sensor['valor'] = 0
                            print(f"--- RESETANDO CONTADOR DE TOSSE PARA O NOVO DIA: {today} ---")
                    LAST_COUGH_RESET_DAY = today
                
                # Criar dicionário com valores atuais dos sensores
                sensor_dict = {}
                for col in DB_COLUMNS[2:]:  # Exclui 'Data' e 'Hora'
                    found = False
                    for sensor in lista_sensores:
                        if sensor['id'] == col:
                            sensor_dict[col] = sensor['valor']
                            found = True
                            break
                    if not found:
                        sensor_dict[col] = 0  # Valor padrão se sensor não encontrado
                        print(f"--- Aviso: Sensor '{col}' não encontrado em lista_sensores, usando 0 ---")

                # Garantir que contagem-tosse seja consistente
                sensor_dict['contagem-tosse'] = COUGH_COUNT

                agora = datetime.now()
                values = [
                    agora.strftime("%Y-%m-%d"),
                    agora.strftime("%H:%M:%S")
                ] + [float(sensor_dict.get(col, 0)) for col in DB_COLUMNS[2:]]
                
                # Log para depuração
                for col, val in zip(DB_COLUMNS[2:], values[2:]):
                    if val == 0 and col != 'contagem-tosse':
                        print(f"--- Aviso: Salvando valor 0 para '{col}' ---")

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(DB_COLUMNS))
            cursor.execute(f"INSERT INTO sensores VALUES ({placeholders})", values)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Erro no loop de salvar DB: {e}")
        time.sleep(10)

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Erro ao conectar ao broker MQTT: {e}")

def incrementa_contador_tosse():
    global COUGH_COUNT
    with sensor_lock:
        COUGH_COUNT += 1
        for sensor in lista_sensores:
            if sensor['id'] == 'contagem-tosse':
                sensor['valor'] = COUGH_COUNT
                print(f"--- CONTADOR DE TOSSE INCREMENTADO: {COUGH_COUNT} ---")
                return

def iniciar_detector_tosse():
    # CORREÇÃO: Todo o bloco de código abaixo foi indentado com 4 espaços
    try:
        print("Carregando modelo de detecção de tosse...")

        # Verifica se os arquivos realmente existem antes de tentar carregá-los
        if not os.path.exists(MODEL_FILENAME):
            print(f"ERRO: Arquivo do modelo não encontrado em: {MODEL_FILENAME}")
            return False # MELHORIA: Usa 'return' em vez de 'exit()'

        if not os.path.exists(SCALER_FILENAME):
            print(f"ERRO: Arquivo do scaler não encontrado em: {SCALER_FILENAME}")
            return False # MELHORIA: Usa 'return' em vez de 'exit()'

        # Monta os caminhos para os arquivos do modelo
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        
        print("Modelo e scaler carregados. Iniciando escuta...")
        live_cough_counter(model, scaler, update_callback=incrementa_contador_tosse)
        
        return True # Retorna sucesso se a função terminar normalmente

    except Exception as e:
        # Este bloco vai capturar o erro 'invalid load key' e outros problemas
        print(f"ERRO ao iniciar o detector de tosse: {e}")
        print("O arquivo pode estar corrompido ou inacessível. Tente retreinar o modelo.")
        return False # MELHORIA: Usa 'return' em vez de 'exit()'

# Thread para MQTT e DB
threading.Thread(target=salvar_dados_db, daemon=True).start()
threading.Thread(target=start_mqtt, daemon=True).start()
threading.Thread(target=iniciar_detector_tosse, daemon=True).start()

def atualizar_sensores_por_ia(dados_ia):
    for sensor_id, valor in dados_ia.items():
        atualizar_sensor(sensor_id, valor)

@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

@sensores_bp.route('/api/sensores')
def api_sensores():
    with sensor_lock:
        return jsonify(lista_sensores)