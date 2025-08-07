from flask import Blueprint, render_template, jsonify
import paho.mqtt.client as mqtt
import json
import threading
import os
from datetime import datetime
import csv
import time
from routes.api import get_weather, get_air_quality, get_user_location
import pickle 
from routes.cough_detector import live_cough_counter, classify_cough, features

sensores_bp = Blueprint('sensores', __name__)

# Configurações
CSV_PATH = r'E:\Dev\TCC-asma\ia\dados\sensores.csv'
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
sensor_lock = threading.Lock()

# Colunas e Sensores Padrão 
CSV_COLUMNS = [
    'Data', 'Hora', 'frequencia-respiratoria', 'batimentos-cardiacos', 'saturacao',
    'temperatura-corporal', 'temperatura-ambiente', 'temperatura-oximetro', 'qualidade-ar-pm25',
    'qualidade-ar-pm10', 'qualidade-ar-aqi', 'movimento-toracico', 'contagem-tosse','som',
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
    {"id": "movimento-toracico", "valor": 0, "unidade": "Hz"},
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
    for s in sensores:
        if s['id'] == 'umidade':
            s['valor'] = humidity
        elif s['id'] == 'qualidade-ar-pm25':
            s['valor'] = float(pm2_5)
        elif s['id'] == 'qualidade-ar-pm10':
            s['valor'] = float(pm10)
        elif s['id'] == 'qualidade-ar-aqi':
            s['valor'] = float(aqi)
    return sensores


lista_sensores = inicializar_sensores()

# MQTT e salvar CSV 
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
                sensor['valor'] = valor
                print(f"--- Sensor '{sensor_id}' atualizado para: {valor} ---") # Log de atualização
                return
        unidade = UNIDADES.get(sensor_id, '')
        lista_sensores.append({'id': sensor_id, 'valor': valor, 'unidade': unidade})
        print(f"--- Sensor novo '{sensor_id}' adicionado com valor: {valor} ---")


def on_message(client, userdata, msg):
    try:
        sensor_id = msg.topic.split('/')[-1]
        payload = msg.payload.decode('utf-8')
        print(f"Recebido no tópico {msg.topic}: '{payload}'")
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

def salvar_dados_csv():
    time.sleep(10)
    while True:
        try:
            with sensor_lock:
                sensor_dict = {s['id']: s['valor'] for s in lista_sensores}
                agora = datetime.now()
                linha = [
                    agora.strftime("%Y-%m-%d"),
                    agora.strftime("%H:%M:%S")
                ] + [sensor_dict.get(col, 0) for col in CSV_COLUMNS[2:]]
            arquivo_existe = os.path.isfile(CSV_PATH)
            try:
                with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    if not arquivo_existe:
                        writer.writerow(CSV_COLUMNS)
                    writer.writerow(linha)
                # print(f"Dados salvos em {CSV_PATH} com sucesso.") # Descomente se quiser log frequente
            except PermissionError:
                print(f"Erro de permissão ao salvar em {CSV_PATH}.")
            except IOError as e:
                print(f"Erro de E/S ao salvar CSV: {e}")
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")
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

# --- NOVA SEÇÃO: INTEGRAÇÃO DA DETECÇÃO DE TOSSE ---
def incrementa_contador_tosse():
    """Função de callback que incrementa o valor do sensor de tosse."""
    with sensor_lock:
        for sensor in lista_sensores:
            if sensor['id'] == 'contagem-tosse':
                sensor['valor'] += 1
                print(f"--- CONTADOR DE TOSSE INCREMENTADO: {sensor['valor']} ---")
                return

def iniciar_detector_tosse():
    """Carrega o modelo e inicia a detecção de tosse em loop."""
    try:
        print("Carregando modelo de detecção de tosse...")
        model_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classifier'
        scaler_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classification_scaler'
        
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        print("Modelo carregado. Iniciando escuta...")
        # A função live_cough_counter rodará indefinidamente, passando 'incrementa_contador_tosse' como callback
        live_cough_counter(model, scaler, update_callback=incrementa_contador_tosse)

    except FileNotFoundError:
        print("ERRO: Arquivos de modelo ou scaler não encontrados. A detecção de tosse não será iniciada.")
    except Exception as e:
        print(f"ERRO ao iniciar o detector de tosse: {e}")

# Thread para MQTT e CSV
threading.Thread(target=salvar_dados_csv, daemon=True).start()
threading.Thread(target=start_mqtt, daemon=True).start()
# --- INICIAR A NOVA THREAD PARA DETECÇÃO DE TOSSE ---
threading.Thread(target=iniciar_detector_tosse, daemon=True).start()

# Função para integração com IA própria
def atualizar_sensores_por_ia(dados_ia):
    for sensor_id, valor in dados_ia.items():
        atualizar_sensor(sensor_id, valor)

# Rotas Flask (sem alteração)
@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

@sensores_bp.route('/api/sensores')
def api_sensores():
    with sensor_lock:
        return jsonify(lista_sensores)