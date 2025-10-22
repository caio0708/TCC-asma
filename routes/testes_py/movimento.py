import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime

# Configuração da porta serial
# Lembre-se de ajustar a porta COM conforme necessário
ser = serial.Serial('COM6', 115200, timeout=1) 

# Buffers para dados (mantém últimos 200 pontos)
max_points = 200
timestamps = deque(maxlen=max_points)
accel_x = deque(maxlen=max_points)
accel_y = deque(maxlen=max_points)
accel_z = deque(maxlen=max_points)
magnitude = deque(maxlen=max_points)
states = deque(maxlen=max_points)

# Estados para coloração
STATE_COLORS = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
STATE_NAMES = {0: 'Repouso', 1: 'Atividade Leve', 
               2: 'Atividade Moderada', 3: 'Tosse'}

# Configuração dos gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Monitoramento de Atividades - MPU6050', fontsize=16)

def update_plot(frame):
    # Lê dados da serial
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        
        # Ignora mensagens de debug
        if line.startswith('TOSSE') or line.startswith('ATIVIDADE'):
            print(f"[EVENTO] {line}")
            return
        
        try:
            # Parse: timestamp,ax,ay,az,magnitude,estado
            data = line.split(',')
            if len(data) == 6:
                timestamp = float(data[0]) / 1000  # converte para segundos
                ax = float(data[1])
                ay = float(data[2])
                az = float(data[3])
                mag = float(data[4])
                state = int(data[5])
                
                # Adiciona aos buffers
                timestamps.append(datetime.now())
                accel_x.append(ax)
                accel_y.append(ay)
                accel_z.append(az)
                magnitude.append(mag)
                states.append(state)
        except (ValueError, IndexError):
            # Ignora linhas mal formatadas
            pass
    
    # Limpa gráficos
    ax1.clear()
    ax2.clear()
    
    # Gráfico 1: Acelerações nos 3 eixos
    if len(timestamps) > 0:
        ax1.plot(timestamps, accel_x, 'r-', label='Accel X', linewidth=1)
        ax1.plot(timestamps, accel_y, 'g-', label='Accel Y', linewidth=1)
        ax1.plot(timestamps, accel_z, 'b-', label='Accel Z', linewidth=1)
        ax1.set_ylabel('Aceleração (m/s²)')
        ax1.set_title('Acelerações por Eixo')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Magnitude com código de cores por estado
    if len(timestamps) > 1: # Precisa de pelo menos 2 pontos para desenhar
        # Plota magnitude com cores baseadas no estado
        for i in range(len(timestamps)-1):
            color = STATE_COLORS.get(states[i], 'gray')
            # CORREÇÃO APLICADA AQUI: Cria uma lista com os dois pontos do segmento
            ax2.plot([timestamps[i], timestamps[i+1]], [magnitude[i], magnitude[i+1]], 
                     color=color, linewidth=2)
        
        # Linhas de threshold
        ax2.axhline(y=2.5, color='red', linestyle='--', label='Threshold Tosse', alpha=0.5)
        ax2.axhline(y=0.3, color='orange', linestyle='--', label='Threshold Atividade', alpha=0.5)
        ax2.axhline(y=0.15, color='green', linestyle='--', label='Threshold Repouso', alpha=0.5)
        
        ax2.set_xlabel('Tempo')
        ax2.set_ylabel('Magnitude (g)')
        ax2.set_title('Magnitude da Aceleração (sem gravidade)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Adiciona estado atual no canto
        if len(states) > 0:
            current_state = states[-1]
            state_text = STATE_NAMES.get(current_state, 'Desconhecido')
            ax2.text(0.02, 0.98, f'Estado: {state_text}', 
                     transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', 
                               facecolor=STATE_COLORS.get(current_state, 'gray'), 
                               alpha=0.7))
    
    # Formata eixo X para mostrar apenas horas
    fig.autofmt_xdate()
    # Ajusta o layout para evitar sobreposição
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste para o título principal

# Inicia animação
# CORREÇÃO DO WARNING: adicionado cache_frame_data=False
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
plt.show()

# Fecha porta serial ao terminar
print("Fechando porta serial.")
ser.close()