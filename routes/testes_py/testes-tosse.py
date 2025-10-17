# -*- coding: utf-8 -*-
"""
Detector de Tosse em Tempo Real com YAMNet

Este script utiliza o microfone para capturar áudio ao vivo, classifica-o
usando o modelo pré-treinado YAMNet e conta o número de tosses detetadas.

NOTA: Este script requer a biblioteca 'requests'. Se não a tiver, instale-a com:
pip install requests
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import csv
import io
import requests # Importa a biblioteca para fazer pedidos HTTP

# --- Parâmetros de Configuração ---

# URL do modelo YAMNet no TensorFlow Hub
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
# URL direta para o ficheiro CSV do mapa de classes do YAMNet
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


# Limiar de confiança para a deteção (ajuste entre 0.1 e 0.9)
# Um valor mais alto torna a deteção mais rigorosa (menos falsos positivos).
DETECTION_THRESHOLD = 0.2

# Duração do cooldown em segundos para evitar contagens múltiplas da mesma tosse
COOLDOWN_SECONDS = 1.0

# --- Carregamento do Modelo e Metadados ---

print("A carregar o modelo YAMNet...")
try:
    # Carrega o modelo a partir do Hub
    model = hub.load(YAMNET_MODEL_URL)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique a sua ligação à internet ou a URL do modelo.")
    exit()

def find_cough_class_index(class_map_csv_text):
    """
    Analisa o CSV do mapa de classes para encontrar o índice da classe 'Cough'.
    Retorna o índice numérico ou -1 se não for encontrado.
    """
    # O identificador de máquina (MID) para 'Cough' no AudioSet
    COUGH_MID = '/m/01b_21'
    with io.StringIO(class_map_csv_text) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Pula o cabeçalho
        for row in reader:
            # A estrutura do CSV é: index, mid, display_name
            # CORREÇÃO: O ID (mid) está na segunda coluna (índice 1), não na terceira.
            if len(row) >= 2 and row[1] == COUGH_MID:
                return int(row[0])
    return -1

# --- Descarregar o mapa de classes diretamente da internet ---
print("A descarregar o mapa de classes...")
try:
    response = requests.get(CLASS_MAP_URL)
    response.raise_for_status()  # Verifica se o download foi bem-sucedido
    class_map_csv_text = response.text
    print("Mapa de classes descarregado com sucesso.")
except requests.exceptions.RequestException as e:
    print(f"Erro ao descarregar o mapa de classes: {e}")
    print("Verifique a sua ligação à internet.")
    exit()

# --- Encontrar o índice da classe 'Cough' de forma robusta pelo seu ID ---
COUGH_CLASS_INDEX = find_cough_class_index(class_map_csv_text)

if COUGH_CLASS_INDEX == -1:
    print("Erro: O ID da classe 'Cough' (/m/01b_21) não foi encontrado no mapa de classes descarregado.")
    exit()
else:
    print(f"Classe 'Cough' encontrada no índice: {COUGH_CLASS_INDEX}")


# --- Variáveis Globais para o Estado da Deteção ---
cough_count = 0
cooldown_counter = 0

# --- Lógica de Processamento de Áudio ---

def audio_callback(indata, frames, time, status):
    """
    Esta função é chamada para cada novo bloco de áudio do microfone.
    """
    global cough_count, cooldown_counter

    if status:
        print(status, flush=True)

    # Converte os dados de áudio para o formato esperado pelo YAMNet (float32, -1 a 1)
    audio_data = indata[:, 0]
    
    # Executa a inferência do modelo
    scores, embeddings, spectrogram = model(audio_data)
    
    # Extrai as pontuações e converte para um array numpy
    scores_np = scores.numpy()
    
    # A pontuação para a classe 'Cough' é a média das pontuações ao longo do tempo para esse índice
    cough_scores = scores_np[:, COUGH_CLASS_INDEX]
    mean_cough_score = np.mean(cough_scores)

    # Lógica de deteção e cooldown
    if cooldown_counter > 0:
        cooldown_counter -= 1
    
    if mean_cough_score > DETECTION_THRESHOLD and cooldown_counter == 0:
        cough_count += 1
        print(f"TOSSE DETETADA! | Pontuação: {mean_cough_score:.2f} | Contagem Total: {cough_count}")
        
        # Ativa o cooldown para evitar deteções repetidas
        sample_rate = sd.query_devices(None, 'input')['default_samplerate']
        cooldown_frames = int(COOLDOWN_SECONDS * (sample_rate / frames))
        cooldown_counter = cooldown_frames

# --- Execução Principal ---

def main():
    print("\n--- Detector de Tosse em Tempo Real ---")
    print("A escutar o microfone... Pressione Ctrl+C para parar.")
    
    try:
        # O YAMNet espera uma taxa de amostragem de 16kHz
        sample_rate = 16000
        
        # O YAMNet processa blocos de 0.975 segundos
        block_duration = 0.975
        block_size = int(sample_rate * block_duration)

        # Configura e inicia o stream de áudio do microfone
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            callback=audio_callback,
            dtype='float32',
            blocksize=block_size
        ):
            while True:
                # Mantém o script a correr enquanto o stream está ativo
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo utilizador.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        print("Verifique se tem um microfone ligado e se as permissões estão corretas.")
    finally:
        print("--- Fim da Sessão ---")
        print(f"Contagem final de tosses: {cough_count}")

if __name__ == "__main__":
    main()