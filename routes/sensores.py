# sensores.py

from flask import Blueprint, render_template, jsonify
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
    api_key = os.getenv('API_WEATHER_KEY')
    temp_api, humidity_api = get_weather(lat, lon)
    aqi, pm2_5, pm10, o3, no2, so2 = get_air_quality(lat, lon, api_key)
    sensores = [dict(s) for s in SENSORES_PADRAO]
    with sensor_lock:
        for s in sensores:
            if s['id'] == 'umidade':
                s['valor'] = float(humidity_api) if humidity_api is not None else 0
            elif s['id'] == 'qualidade-ar-pm25':
                s['valor'] = float(pm2_5) if pm2_5 is not None else 0
            elif s['id'] == 'qualidade-ar-pm10':
                s['valor'] = float(pm10) if pm10 is not None else 0
            elif s['id'] == 'qualidade-ar-aqi':
                s['valor'] = float(aqi) if aqi is not None else 0
            elif s['id'] == 'qualidade-ar-o3':
                s['valor'] = float(o3) if aqi is not None else 0
            elif s['id'] == 'qualidade-ar-no2':
                s['valor'] = float(no2) if aqi is not None else 0
            elif s['id'] == 'qualidade-ar-so2':
                s['valor'] = float(so2) if aqi is not None else 0                
            elif s['id'] == 'temperatura-ambiente':
                s['valor'] = float(temp_api) if temp_api is not None else 0
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

# ===== FUNÇÃO DE SALVAR NO DB (MODIFICADA) =====
def salvar_dados_db(app_state, state_lock):
    """
    Salva os dados do estado central da aplicação no banco de dados periodicamente.
    Agora recebe 'app_state' e 'state_lock' de app.py.
    """
    time.sleep(10) # Espera inicial
    while True:
        try:
            # Pega uma cópia segura do estado atual
            with state_lock:
                current_state = app_state.copy()

            # Pega o total de tosses do dia no DB para ter o valor cumulativo correto
            db_cough_count = get_today_cough_count_from_db()
            
            # A contagem correta é o valor já salvo no DB mais o que foi contado nesta sessão
            # (Isso evita zerar a contagem se o app reiniciar)
            # A forma mais simples é pegar o máximo do que já existe no dia
            final_cough_count = max(db_cough_count, current_state.get('coughs_total', 0))

            # Cria o dicionário para salvar no DB
            sensor_dict = {}
            for col in DB_COLUMNS[2:]:
                # Usa o valor do estado central se existir, senão usa 0
                sensor_dict[col] = current_state.get(col, 0)
            
            # Atualiza com o valor de tosse correto e outros dados importantes
            sensor_dict['contagem-tosse'] = final_cough_count
            sensor_dict['saturacao'] = current_state.get('spo2') or 0
            sensor_dict['batimentos-cardiacos'] = current_state.get('bpm') or 0
            sensor_dict['temperatura-ambiente'] = current_state.get('temp') or 0
            sensor_dict['umidade'] = current_state.get('humidity') or 0
            
            # Garantir que todos os valores sejam numéricos antes de salvar
            for key, value in sensor_dict.items():
                if value is None:
                    sensor_dict[key] = 0
                # Tenta converter para float, se falhar, usa 0.
                try:
                    # A conversão para float já acontece na linha abaixo, então esta é redundante
                    # mas serve como uma verificação explícita.
                    sensor_dict[key] = float(sensor_dict[key])
                except (ValueError, TypeError):
                    sensor_dict[key] = 0

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            now = datetime.now()
            data_atual = now.strftime("%Y-%m-%d")
            hora_atual = now.strftime("%H:%M:%S")
            
            # Monta a lista de valores na ordem correta das colunas
            valores_para_salvar = [data_atual, hora_atual] + [sensor_dict.get(col, 0) for col in DB_COLUMNS[2:]]
            
            placeholders = ', '.join(['?'] * len(DB_COLUMNS))
            colunas_str = ', '.join([f'"{col}"' for col in DB_COLUMNS])
            
            sql = f"INSERT INTO sensores ({colunas_str}) VALUES ({placeholders})"
            
            cursor.execute(sql, valores_para_salvar)
            conn.commit()
            conn.close()
            
            # print(f"Dados salvos no DB em {hora_atual}")

        except Exception as e:
            print(f"Erro no loop de salvar DB: {e}")
        
        time.sleep(30) # Salva a cada 30 segundos

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Erro ao conectar ao broker MQTT: {e}")

# Thread para MQTT
threading.Thread(target=start_mqtt, daemon=True).start()

def atualizar_sensores_por_ia(dados_ia):
    for sensor_id, valor in dados_ia.items():
        atualizar_sensor(sensor_id, valor)

@sensores_bp.route('/sensores')
def sensores():
    return render_template('sensores.html')

@sensores_bp.route('/api/sensores')
def api_sensores():
    # A lógica para buscar e sincronizar a contagem de tosse permanece a mesma
    db_cough_count = get_today_cough_count_from_db()

    data_formatada = []
    with sensor_lock:
        # Garante que o valor de tosse está atualizado na lista principal
        cough_sensor_found = False
        for sensor in lista_sensores:
            if sensor['id'] == 'contagem-tosse':
                sensor['valor'] = max(sensor.get('valor', 0), db_cough_count)
                cough_sensor_found = True
                break
        if not cough_sensor_found:
            lista_sensores.append({'id': 'contagem-tosse', 'valor': db_cough_count, 'unidade': 'no dia'})
        
        # Itera sobre a lista de sensores para formatar os valores para a resposta JSON
        for sensor in lista_sensores:
            sensor_copia = sensor.copy()
            valor = sensor_copia.get('valor')

            # Verifica se o valor é numérico antes de tentar formatar
            if isinstance(valor, (int, float)):
                sensor_id = sensor_copia.get('id')

                # Batimentos Cardíacos: arredonda para o inteiro mais próximo (0 casas decimais)
                if sensor_id == 'batimentos-cardiacos':
                    sensor_copia['valor'] = round(valor)
                
                # Saturação e Temperaturas: arredonda para 1 casa decimal
                elif sensor_id in ['saturacao', 'temperatura-corporal', 'temperatura-ambiente', 'temperatura-oximetro', 'frequencia-respiratoria']:
                    sensor_copia['valor'] = round(valor, 1)

            data_formatada.append(sensor_copia)

    return jsonify(data_formatada)