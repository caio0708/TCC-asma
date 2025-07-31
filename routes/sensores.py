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

# Caminho para o arquivo CSV
CSV_PATH = os.path.join('dados', 'sensores.csv')

# Garante que a pasta existe
os.makedirs('dados', exist_ok=True)
sensor_lock = threading.Lock()

lat, lon, city = get_user_location()
API_KEY = '7288a386509b40eb0513fd8500bd5d5d' 

temp, humidity = get_weather(lat, lon)
aqi, pm2_5, pm10 = get_air_quality(lat, lon, API_KEY) 

aqi = float(aqi)
pm2_5 = float(pm2_5)
pm10 = float(pm10)

# Lista inicial de sensores
lista_sensores = [
    {"id": "frequencia-respiratoria", "valor": 0, "unidade": "rpm"},
    {"id": "batimentos-cardiacos",     "valor": 0, "unidade": "bpm"},
    {"id": "saturacao",                "valor": 0, "unidade": "%"},
    {"id": "temperatura-corporal",     "valor": 0, "unidade": "°C"},
    {"id": "temperatura-ambiente",     "valor": 0, "unidade": "°C"},
    {"id": "temperatura-oxi",          "valor": 0, "unidade": "°C"},
    {"id": "qualidade-ar-pm25",        "valor": pm2_5, "unidade": "µg/m³"},
    {"id": "qualidade-ar-pm10",        "valor": pm10, "unidade": "µg/m³"},
    {"id": "qualidade-ar-aqi",         "valor": aqi, "unidade": ""},
    {"id": "movimento-toracico",       "valor": 0, "unidade": "Hz"},
    {"id": "contagem-tosse",           "valor": 0, "unidade": "no dia"},
    {"id": "umidade",                  "valor": humidity, "unidade": "%"},
    {"id": "acelerometro-x",           "valor": 0, "unidade": ""},
    {"id": "acelerometro-y",           "valor": 0, "unidade": ""},
    {"id": "acelerometro-z",           "valor": 0, "unidade": ""},
    {"id": "giroscopio-x",           "valor": 0, "unidade": ""},
    {"id": "giroscopio-y",           "valor": 0, "unidade": ""},
    {"id": "giroscopio-z",           "valor": 0, "unidade": ""}
]

# Unidade por sensor
UNIDADES = {
    "frequencia-respiratoria": "rpm",
    "batimentos-cardiacos":     "bpm",
    "saturacao":                "%",
    "temperatura-corporal":     "°C",
    "temperatura-ambiente":     "°C",
    "temperatura-oxi":          "°C",
    "qualidade-ar-pm25":        "µg/m³",
    "qualidade-ar-pm10":        "µg/m³",
    "qualidade-ar-aqi":         "%",
    "movimento-toracico":       "Hz",
    "umidade":                  "%",
    "teste":                  "%",
    # adicione mais sensores aqui se necessário
}

# Configurações do MQTT
BROKER = "broker.hivemq.com"  # Broker público para testes
PORT = 1883
TOPIC_ALL = "sensorestcc/+"  # Tópico MQTT onde os dados serão recebidos ; sensores/temperatura  …e assim por diante, qualquer coisa no lugar do +.

# Função chamada quando conectado ao broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado ao broker MQTT")
        client.subscribe(TOPIC_ALL)  # Inscreve no tópico
    else:
        print(f"Falha na conexão, código: {rc}")

# Callback de mensagem: atualiza apenas o sensor específico
def on_message(client, userdata, msg):
    global lista_sensores
    try:
        # Extrai sensor_id do tópico 'sensores/<sensor_id>'
        sensor_id = msg.topic.split('/')[-1]
        payload = msg.payload.decode('utf-8')
        print(f"Recebido no tópico {msg.topic}: '{payload}'")

        # Decodifica JSON ou tenta converter para número simples
        valor = None
        try:
            dados = json.loads(payload)
            if isinstance(dados, dict) and 'valor' in dados:
                valor = dados['valor']
            else:
                valor = dados
        except json.JSONDecodeError:
            # Payload simples: tenta converter
            try:
                valor = float(payload) if '.' in payload else int(payload)
            except ValueError:
                valor = payload  # string genérica

        # Atualiza sensor existente ou adiciona novo
        for sensor in lista_sensores:
            if sensor['id'] == sensor_id:
                sensor['valor'] = valor
                break
        else:
            # Se não existir, adiciona novo sensor
            unidade = UNIDADES.get(sensor_id, '')
            lista_sensores.append({
                'id': sensor_id,
                'valor': valor,
                'unidade': unidade
            })

    except Exception as e:
        print(f"Erro ao processar mensagem MQTT: {e}")

# Função que salva os dados dos sensores em um CSV
def salvar_dados_csv():
    time.sleep(10)  # Aguarda para garantir que dados cheguem
    while True:
        try:
            with sensor_lock:
                cabecalho = ['Data', 'Hora'] + [s['id'] for s in lista_sensores]
                agora = datetime.now()
                data_str = agora.strftime("%Y-%m-%d")
                hora_str = agora.strftime("%H:%M:%S")
                linha = [data_str, hora_str] + [s['valor'] for s in lista_sensores]

            arquivo_existe = os.path.isfile(CSV_PATH)

            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not arquivo_existe:
                    writer.writerow(cabecalho)
                writer.writerow(linha)

            print("Dados salvos no CSV.")

        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")

        time.sleep(300) # Espera 5 minutos (300 segundos)

# Inicia a thread de salvamento periódico
salvamento_thread = threading.Thread(target=salvar_dados_csv, daemon=True)
salvamento_thread.start()

# Configuração do cliente MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Função para iniciar o cliente MQTT em uma thread separada
def start_mqtt():
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# Inicia o MQTT em uma thread separada ao carregar o blueprint
mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

# Página de visualização dos sensores
@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

# API que fornece os valores dos sensores
@sensores_bp.route('/api/sensores')
def api_sensores():
    return jsonify(lista_sensores)

