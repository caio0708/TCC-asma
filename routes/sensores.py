from flask import Blueprint, render_template, jsonify
import paho.mqtt.client as mqtt
import json
import threading
import os
from datetime import datetime
import csv
import time
from routes.api import get_weather, get_air_quality, get_user_location

sensores_bp = Blueprint('sensores', __name__)

# Configurações
CSV_PATH = os.path.join('dados', 'sensores.csv')
os.makedirs('dados', exist_ok=True)
sensor_lock = threading.Lock()

# Sensores e unidades
SENSORES_PADRAO = [
    {"id": "frequencia-respiratoria", "valor": 0, "unidade": "rpm"},
    {"id": "batimentos-cardiacos",     "valor": 0, "unidade": "bpm"},
    {"id": "saturacao",                "valor": 0, "unidade": "%"},
    {"id": "temperatura-corporal",     "valor": 0, "unidade": "°C"},
    {"id": "temperatura-ambiente",     "valor": 0, "unidade": "°C"},
    {"id": "temperatura-oxi",          "valor": 0, "unidade": "°C"},
    {"id": "qualidade-ar-pm25",        "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-pm10",        "valor": 0, "unidade": "µg/m³"},
    {"id": "qualidade-ar-aqi",         "valor": 0, "unidade": ""},
    {"id": "movimento-toracico",       "valor": 0, "unidade": "Hz"},
    {"id": "contagem-tosse",           "valor": 0, "unidade": "no dia"},
    {"id": "umidade",                  "valor": 0, "unidade": "%"},
    {"id": "acelerometro-x",           "valor": 0, "unidade": ""},
    {"id": "acelerometro-y",           "valor": 0, "unidade": ""},
    {"id": "acelerometro-z",           "valor": 0, "unidade": ""},
    {"id": "giroscopio-x",             "valor": 0, "unidade": ""},
    {"id": "giroscopio-y",             "valor": 0, "unidade": ""},
    {"id": "giroscopio-z",             "valor": 0, "unidade": ""}
]
UNIDADES = {s['id']: s['unidade'] for s in SENSORES_PADRAO}

# Inicialização dos sensores
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

# MQTT Configuração
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
                return
        # Novo sensor
        unidade = UNIDADES.get(sensor_id, '')
        lista_sensores.append({'id': sensor_id, 'valor': valor, 'unidade': unidade})

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
                cabecalho = ['Data', 'Hora'] + [s['id'] for s in lista_sensores]
                agora = datetime.now()
                linha = [agora.strftime("%Y-%m-%d"), agora.strftime("%H:%M:%S")] + [s['valor'] for s in lista_sensores]
            arquivo_existe = os.path.isfile(CSV_PATH)
            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not arquivo_existe:
                    writer.writerow(cabecalho)
                writer.writerow(linha)
            print("Dados salvos no CSV.")
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")
        time.sleep(300)

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# Thread para MQTT e CSV
threading.Thread(target=salvar_dados_csv, daemon=True).start()
threading.Thread(target=start_mqtt, daemon=True).start()

# Função para integração com IA própria
def atualizar_sensores_por_ia(dados_ia):
    """
    Atualiza sensores com dados vindos da IA própria.
    Exemplo de dados_ia: {'batimentos-cardiacos': 80, 'temperatura-corporal': 36.5}
    """
    for sensor_id, valor in dados_ia.items():
        atualizar_sensor(sensor_id, valor)

# Rotas Flask
@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

@sensores_bp.route('/api/sensores')
def api_sensores():
    with sensor_lock:
        return jsonify(lista_sensores)