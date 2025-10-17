# app.py
import os
import threading
import asyncio 
import paho.mqtt.client as mqtt
from flask import Flask, jsonify

# ===== REGISTRO DAS ROTAS (BLUEPRINTS) =====
from routes.painel import painel_bp
from routes.sensores import sensores_bp, salvar_dados_db 
from routes.chat import chat_bp
from routes.insights import insights_bp
from routes.configuracoes import configuracoes_bp

# ===== MÓDULOS DE BACKGROUND =====
from routes.max30102_MQTT import iniciar_monitor_sensores  # max30102_BLUETOOTH
from routes.cough_detector import iniciar_detector_tosse

# =============================================================================
# ESTADO GLOBAL E CENTRALIZADO DA APLICAÇÃO
# =============================================================================
app_state = {
    'spo2': 0, 'bpm': 0, 'temp': 0, 'humidity': 0, 'quality': 0.0,
    'connected': False, 'coughs_total': 0
}
state_lock = threading.Lock()

# =============================================================================
# CONFIGURAÇÃO DO MQTT (sem alterações)
# =============================================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "sensorestcc"
MQTT_CLIENT_ID = f"flask-app-monitor-{os.getpid()}"

def setup_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] ✅ Conectado ao Broker MQTT!")
        else:
            print(f"[MQTT] ❌ Falha ao conectar, código: {rc}\n")
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
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
    """Callback para quando dados do oxímetro/temp são recebidos."""
    global app_state, mqtt_client
    with state_lock:
        app_state.update(data)
        spo2_val = data.get('spo2') or 0
        bpm_val = data.get('bpm') or 0
        temp_val = data.get('temp') or 0
        hum_val = data.get('humidity') or 0
    if mqtt_client:
        mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/saturacao", f"{spo2_val:.1f}")
        mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/batimentos-cardiacos", f"{bpm_val:.0f}")
        mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/temperatura-ambiente", f"{temp_val:.1f}")
        mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/umidade", f"{hum_val:.1f}")

def on_cough_detected():
    """Callback para quando uma tosse é detectada."""
    global app_state, mqtt_client
    with state_lock:
        # A contagem de tosse é cumulativa na sessão atual
        app_state['coughs_total'] += 1
        cough_count = app_state['coughs_total']
        print(f"[App] Tosse registrada! Total na sessão: {cough_count}")
    if mqtt_client:
        # Publica o novo total de tosses
        mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/contagem-tosse", str(cough_count))

# =============================================================================
# CONFIGURAÇÃO DA APLICAÇÃO FLASK
# =============================================================================
app = Flask(__name__)
app.secret_key = 'uma-senha-muito-secreta-e-complexa'

# Registra os blueprints
app.register_blueprint(sensores_bp)
app.register_blueprint(painel_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(insights_bp)
app.register_blueprint(configuracoes_bp)

# API para fornecer o estado centralizado ao front-end
@app.route('/api/dados-em-tempo-real')
def get_realtime_data():
    with state_lock:
        # Retorna uma cópia do estado atual
        return jsonify(app_state)

# =============================================================================
# FUNÇÃO AUXILIAR PARA EXECUTAR CÓDIGO ASSÍNCRONO EM UMA THREAD
# =============================================================================
def run_async_in_thread(async_func, *args):
    """
    Cria um novo loop de eventos asyncio e executa a corrotina nele.
    Essencial para rodar o bleak (async) dentro de uma thread síncrona.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func(*args))
    loop.close()

# =============================================================================
# INICIALIZAÇÃO DOS PROCESSOS
# =============================================================================
if __name__ == '__main__':
    print("[App] Iniciando processos de background...")

    # --- Thread do monitor de sensores  ---
    sensor_thread = threading.Thread(
        target=iniciar_monitor_sensores,   # target=iniciar_monitor_sensores, target=run_async_in_thread,
        args=(on_sensor_data_received,),   # args=(on_sensor_data_received,), args=(iniciar_monitor_sensores, on_sensor_data_received),
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


    # --- Inicia o servidor Flask ---
    print("[App] Iniciando servidor Flask...")
    app.run(debug=True, use_reloader=False)