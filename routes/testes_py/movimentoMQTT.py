import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime

# --- CONFIGURAÇÕES DO USUÁRIO ---
# CORREÇÃO: Apontando para o broker público HiveMQ
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
# CORREÇÃO: Tópicos correspondentes ao código do ESP32
DATA_TOPIC = "tcc/mpu6050/data"
EVENT_TOPIC = "sensorestcc/mpu6050/eventos"

# Buffers de dados
max_points = 200
timestamps = deque(maxlen=max_points)
accel_x = deque(maxlen=max_points)
accel_y = deque(maxlen=max_points)
accel_z = deque(maxlen=max_points)
magnitude = deque(maxlen=max_points)
states = deque(maxlen=max_points)

# Mapeamento de estados
STATE_COLORS = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
STATE_NAMES = {0: 'Repouso', 1: 'Atividade Leve', 2: 'Atividade Moderada', 3: 'Tosse'}

# --- FUNÇÕES MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Conectado ao Broker MQTT HiveMQ!")
        client.subscribe(DATA_TOPIC)
        client.subscribe(EVENT_TOPIC)
        print(f"Inscrito no tópico de dados: '{DATA_TOPIC}'")
        print(f"Inscrito no tópico de eventos: '{EVENT_TOPIC}'")
    else:
        print(f"❌ Falha na conexão, código de retorno: {rc}\n")

def on_message(client, userdata, msg):
    """Callback para processar mensagens recebidas."""
    payload = msg.payload.decode('utf-8')
    
    if msg.topic == DATA_TOPIC:
        try:
            data = payload.split(',')
            if len(data) == 6:
                timestamps.append(datetime.now())
                accel_x.append(float(data[1]))
                accel_y.append(float(data[2]))
                accel_z.append(float(data[3]))
                magnitude.append(float(data[4]))
                states.append(int(data[5]))
        except Exception as e:
            print(f"Erro ao processar dados: {e}")
    elif msg.topic == EVENT_TOPIC:
        print(f"--- [EVENTO RECEBIDO] ---> {payload} ---")

# Configuração do cliente MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# --- CONFIGURAÇÃO E ATUALIZAÇÃO DO GRÁFICO (Otimizado) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Monitoramento de Atividades - MPU6050 via MQTT', fontsize=16)

# Cria as linhas uma vez para otimização
line_ax, = ax1.plot([], [], 'r-', label='Accel X', linewidth=1)
line_ay, = ax1.plot([], [], 'g-', label='Accel Y', linewidth=1)
line_az, = ax1.plot([], [], 'b-', label='Accel Z', linewidth=1)

def init_plot():
    """Configura os elementos estáticos do gráfico."""
    ax1.set_ylabel('Aceleração (m/s²)')
    ax1.set_title('Acelerações por Eixo')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-20, 20)

    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Magnitude (g)')
    ax2.set_title('Magnitude da Aceleração (sem gravidade)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 4.0)
    
    ax2.axhline(y=2.5, color='red', linestyle='--', label='Threshold Tosse', alpha=0.5)
    ax2.axhline(y=0.3, color='orange', linestyle='--', label='Threshold Atividade', alpha=0.5)
    ax2.axhline(y=0.15, color='green', linestyle='--', label='Threshold Repouso', alpha=0.5)
    ax2.legend(loc='upper right')
    return line_ax, line_ay, line_az,

def update_plot(frame):
    """Atualiza os dados do gráfico periodicamente."""
    if not timestamps:
        return line_ax, line_ay, line_az,

    line_ax.set_data(timestamps, accel_x)
    line_ay.set_data(timestamps, accel_y)
    line_az.set_data(timestamps, accel_z)
    ax1.relim()
    ax1.autoscale_view(True, True, True)

    ax2.clear()
    ax2.set_xlabel('Tempo'); ax2.set_ylabel('Magnitude (g)'); ax2.set_title('Magnitude da Aceleração')
    ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.5, 4.0)
    ax2.axhline(y=2.5, color='red', linestyle='--', label='Threshold Tosse', alpha=0.5)
    ax2.axhline(y=0.3, color='orange', linestyle='--', label='Threshold Atividade', alpha=0.5)
    ax2.axhline(y=0.15, color='green', linestyle='--', label='Threshold Repouso', alpha=0.5)
    ax2.legend(loc='upper right')

    for i in range(len(timestamps) - 1):
        color = STATE_COLORS.get(states[i], 'gray')
        ax2.plot([timestamps[i], timestamps[i+1]], [magnitude[i], magnitude[i+1]], color=color, linewidth=2)

    if states:
        current_state = states[-1]
        state_text = STATE_NAMES.get(current_state, 'Desconhecido')
        ax2.text(0.02, 0.98, f'Estado: {state_text}', transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor=STATE_COLORS.get(current_state, 'gray'), alpha=0.7))
    
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return line_ax, line_ay, line_az,

# Inicia a animação
ani = animation.FuncAnimation(fig, update_plot, init_func=init_plot, interval=100, blit=False, cache_frame_data=False)
plt.show()

# Finaliza o loop MQTT
client.loop_stop()
print("Visualização encerrada.")