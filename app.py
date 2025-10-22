# app.py
import os
import threading
import asyncio 
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
import json 

# ===== REGISTRO DAS ROTAS (BLUEPRINTS) =====
from routes.painel import painel_bp
from routes.sensores import sensores_bp, salvar_dados_db 
from routes.chat import chat_bp
from routes.insights import insights_bp
from routes.configuracoes import configuracoes_bp

# ===== MÓDULOS DE BACKGROUND =====
from routes.max30102_MQTT import iniciar_monitor_sensores
from routes.cough_detector import iniciar_detector_tosse
from routes.sensores import salvar_dados_db, atualizar_dados_api_externa

# =============================================================================
# ESTADO GLOBAL E CENTRALIZADO DA APLICAÇÃO
# =============================================================================
app_state = {
    'frequencia-respiratoria': 0, 'batimentos-cardiacos': 0, 'saturacao': 0,
    'temperatura-corporal': 0, 'temperatura-ambiente': 0, 'temperatura-oximetro': 0,
    'qualidade-ar-pm25': 0, 'qualidade-ar-pm10': 0, 'qualidade-ar-aqi': 0,
    'qualidade-ar-o3': 0, 'qualidade-ar-no2': 0, 'qualidade-ar-so2': 0,
    'piezo': 0, 'contagem-tosse': 0, 'som': 0, 'umidade': 0,
    'acelerometro-x': 0, 'acelerometro-y': 0, 'acelerometro-z': 0,
    'giroscopio-x': 0, 'giroscopio-y': 0, 'giroscopio-z': 0,
    'spo2': 0, 'bpm': 0, 
    'connected': False, 'coughs_total': 0
}
state_lock = threading.Lock()

# =============================================================================
# CONFIGURAÇÃO DO MQTT (Centralizado)
# =============================================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_ALL = "sensorestcc/#" # Tópico correto para capturar tudo
MQTT_CLIENT_ID = f"flask-app-monitor-{os.getpid()}"

# Handler de mensagens MQTT unificado
# --- NOVO: alias e mapeamentos tolerantes ao .ino ---
KEY_ALIASES = {
    # vitais
    'hr': 'bpm', 'heart_rate': 'bpm',
    'spo2_pct': 'spo2', 'ox': 'spo2',
    # temperatura/umidade
    'temp': 'temperatura-ambiente', 'tempC': 'temperatura-corporal', 'body_temp': 'temperatura-corporal',
    'ambient_temp': 'temperatura-ambiente', 'ox_temp': 'temperatura-oximetro',
    'hum': 'umidade', 'humidity': 'umidade',
    # ar/qualidade
    'pm25': 'qualidade-ar-pm25', 'pm2_5': 'qualidade-ar-pm25',
    'pm10': 'qualidade-ar-pm10', 'aqi': 'qualidade-ar-aqi',
    'o3': 'qualidade-ar-o3', 'no2': 'qualidade-ar-no2', 'so2': 'qualidade-ar-so2',
    # respiração / som
    'rr': 'frequencia-respiratoria', 'resp_rate': 'frequencia-respiratoria',
    'sound': 'som',
    # IMU
    'accx': 'acelerometro-x', 'accy': 'acelerometro-y', 'accz': 'acelerometro-z',
    'gyrox': 'giroscopio-x', 'gyroy': 'giroscopio-y', 'gyroz': 'giroscopio-z',
    # tosse
    'coughs': 'contagem-tosse'
}

# Quando uma chave primária chega, também atualizamos os “espelhos” usados no UI/DB
RELATED_KEYS = {
    'bpm': ['batimentos-cardiacos'],
    'spo2': ['saturacao'],
}

def _to_float_safe(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().lower()
    if s in ('', 'null', 'none', 'nan', 'inf', '-inf'): return None
    try:
        return float(s.replace(',', '.'))
    except Exception:
        return None

def _apply_update(key, value, app_state, state_lock):
    # normaliza alias
    k = KEY_ALIASES.get(key, key)
    val = _to_float_safe(value)
    if val is None:
        return False
    updated = False
    with state_lock:
        if k in app_state:
            app_state[k] = val
            updated = True
        # atualiza espelhos (ex.: bpm -> batimentos-cardiacos)
        for mirror in RELATED_KEYS.get(k, []):
            if mirror in app_state:
                app_state[mirror] = val
                updated = True
    return updated

def on_message_unified(client, userdata, msg):
    """Processa QUALQUER payload: numérico simples, JSON, ou 'k:v,k:v'."""
    topic = msg.topic
    payload_str = msg.payload.decode('utf-8', errors='ignore').strip()

    try:
        # 1) Tenta JSON
        parsed = None
        try:
            parsed = json.loads(payload_str)
        except json.JSONDecodeError:
            parsed = None

        updated_any = False

        if isinstance(parsed, dict):
            # Caso típico do .ino: um tópico por dispositivo e várias chaves no JSON
            for k, v in parsed.items():
                if isinstance(v, dict):  # ignora sub-objetos (ex.: {unit:"bpm"})
                    continue
                updated_any |= _apply_update(k, v, app_state, state_lock)

        elif parsed is not None:
            # JSON simples (número/string)
            # tenta usar o último segmento do tópico como chave
            sensor_id = topic.split('/')[-1]
            updated_any |= _apply_update(sensor_id, parsed, app_state, state_lock)

        else:
            # 2) Não é JSON: pode ser número puro, ou "k:v,k:v"
            if ':' in payload_str and ',' in payload_str:
                # Ex.: "bpm:72,spo2:98,tempC:36.4"
                pairs = [p for p in payload_str.split(',') if ':' in p]
                for p in pairs:
                    k, v = p.split(':', 1)
                    updated_any |= _apply_update(k.strip(), v.strip(), app_state, state_lock)
            elif ':' in payload_str and '/' not in payload_str:
                # Ex.: "bpm:72"
                k, v = payload_str.split(':', 1)
                updated_any |= _apply_update(k.strip(), v.strip(), app_state, state_lock)
            else:
                # Ex.: tópico termina em 'bpm' e payload "72"
                sensor_id = topic.split('/')[-1]
                updated_any |= _apply_update(sensor_id, payload_str, app_state, state_lock)

        if not updated_any:
            print(f"[MQTT] Aviso: payload ignorado (sem chaves compatíveis): {payload_str} | tópico {topic}")

    except Exception as e:
        print(f"[MQTT] Erro ao processar mensagem: {e} | tópico: {topic} | payload: {payload_str}")

def setup_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] ✅ Conectado ao Broker MQTT!")
            client.subscribe(MQTT_TOPIC_ALL) # Assina o tópico geral
            client.subscribe("sensorestcc/mpu-state")
        else:
            print(f"[MQTT] ❌ Falha ao conectar, código: {rc}\n")
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message_unified # Usa o handler unificado
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"[MQTT] ❌ ERRO: Não foi possível conectar ao broker: {e}")
        return None

mqtt_client = setup_mqtt()

# =============================================================================
# FUNÇÕES DE CALLBACK (Handlers de Eventos)
# =============================================================================
def on_sensor_data_received(data):
    """Callback para quando dados do oxímetro/temp são recebidos (Bluetooth/Serial)."""
    with state_lock:
        app_state['spo2'] = data.get('spo2', app_state['spo2'])
        app_state['saturacao'] = data.get('spo2', app_state['saturacao'])
        app_state['bpm'] = data.get('bpm', app_state['bpm'])
        app_state['batimentos-cardiacos'] = data.get('bpm', app_state['batimentos-cardiacos'])

# Função para executar código assíncrono (se necessário)
def run_async_in_thread(async_func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func(*args))
    loop.close()

def on_cough_detected():
    """Callback para quando uma tosse é detectada."""
    with state_lock:
        app_state['coughs_total'] += 1
        app_state['contagem-tosse'] = app_state['coughs_total']
        print(f"[App] Tosse registrada! Total na sessão: {app_state['coughs_total']}")
    if mqtt_client:
        mqtt_client.publish(f"sensorestcc/contagem-tosse", str(app_state['coughs_total']))

# =============================================================================
# CONFIGURAÇÃO DA APLICAÇÃO FLASK
# =============================================================================
app = Flask(__name__)
app.secret_key = 'uma-senha-muito-secreta-e-complexa'

# Isso permite que a API de sensores acesse os dados em tempo real.
@app.context_processor
def inject_global_state():
    return dict(app_state=app_state, state_lock=state_lock)

app.config['app_state'] = app_state
app.config['state_lock'] = state_lock

# Registra os blueprints
app.register_blueprint(sensores_bp)
app.register_blueprint(painel_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(insights_bp)
app.register_blueprint(configuracoes_bp)

# =============================================================================
# INICIALIZAÇÃO DOS PROCESSOS
# =============================================================================
if __name__ == '__main__':
    print("[App] Iniciando processos de background...")

    # --- Thread do monitor de sensores ---
    sensor_thread = threading.Thread(
        target=run_async_in_thread,
        args=(iniciar_monitor_sensores, on_sensor_data_received),
        daemon=True
    )
    sensor_thread.start()
    print("[App] Thread do monitor de sensores iniciada.")

    # --- Thread do detector de tosse ---
    cough_thread = threading.Thread(
        target=iniciar_detector_tosse,
        args=(on_cough_detected,),
        daemon=True
    )
    cough_thread.start()
    print("[App] Thread do detector de tosse iniciada.")

    # --- Thread para salvar dados no banco de dados ---
    db_saver_thread = threading.Thread(
        target=salvar_dados_db,
        args=(app_state, state_lock), 
        daemon=True
    )
    db_saver_thread.start()
    print("[App] Thread de persistência de dados (DB) iniciada.")

    api_updater_thread = threading.Thread(
            target=atualizar_dados_api_externa,
            args=(app_state, state_lock),
            daemon=True
        )
    api_updater_thread.start()
    print("[App] Thread de atualização de APIs externas iniciada.")

    # --- Inicia o servidor Flask ---
    print("[App] Iniciando servidor Flask...")
    app.run(debug=True, use_reloader=False)