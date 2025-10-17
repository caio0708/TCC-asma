# -*- coding: utf-8 -*-
"""
Detector de Tosse em Tempo Real com YAMNet

Este script utiliza o microfone para capturar áudio ao vivo, classifica-o
usando o modelo pré-treinado YAMNet e notifica a aplicação principal
através de uma função de callback quando uma tosse é detetada.
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import csv
import io
import requests

# --- Parâmetros de Configuração ---
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(YAMNET_MODEL_URL)
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
DETECTION_THRESHOLD = 0.15
COOLDOWN_SECONDS = 1.0 # Aumentado para evitar contagens duplas

# --- Carregamento do Modelo e Metadados ---
print("[Detector de Tosse] A carregar o modelo YAMNet...")
try:
    model = hub.load(YAMNET_MODEL_URL)
    print("[Detector de Tosse] Modelo carregado com sucesso.")
except Exception as e:
    print(f"[Detector de Tosse] Erro ao carregar o modelo: {e}")
    print("[Detector de Tosse] Verifique a sua ligação à internet ou a URL do modelo.")
    exit()

def find_cough_class_index(class_map_csv_text):
    """
    Analisa o CSV do mapa de classes para encontrar o índice da classe 'Cough'.
    """
    COUGH_MID = '/m/01b_21'
    with io.StringIO(class_map_csv_text) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Pula o cabeçalho
        for row in reader:
            if len(row) >= 2 and row[1] == COUGH_MID:
                return int(row[0])
    return -1

print("[Detector de Tosse] A descarregar o mapa de classes...")
try:
    response = requests.get(CLASS_MAP_URL)
    response.raise_for_status()
    class_map_csv_text = response.text
    print("[Detector de Tosse] Mapa de classes descarregado.")
except requests.exceptions.RequestException as e:
    print(f"[Detector de Tosse] Erro ao descarregar o mapa de classes: {e}")
    exit()

COUGH_CLASS_INDEX = find_cough_class_index(class_map_csv_text)

if COUGH_CLASS_INDEX == -1:
    print("[Detector de Tosse] Erro: O ID da classe 'Cough' (/m/01b_21) não foi encontrado.")
    exit()
else:
    print(f"[Detector de Tosse] Classe 'Cough' encontrada no índice: {COUGH_CLASS_INDEX}")


# --- Execução Principal ---

# ALTERAÇÃO PRINCIPAL: A função agora aceita um callback como argumento
def iniciar_detector_tosse(on_cough_detected_callback):
    """
    Inicia a escuta do microfone e chama a função de callback quando uma tosse é detectada.
    
    :param on_cough_detected_callback: A função a ser chamada quando uma tosse é detectada.
    """
    print("\n--- Detector de Tosse em Tempo Real ---")
    print("A escutar o microfone... Pressione Ctrl+C para parar.")

    # Variável para controlar o cooldown
    cooldown_counter = 0

    # Definimos a função de callback de áudio DENTRO da função principal
    # para que ela tenha acesso a `on_cough_detected_callback`
    def audio_callback(indata, frames, time, status):
        nonlocal cooldown_counter # Usamos nonlocal para modificar a variável da função externa

        if status:
            print(f"[Detector de Tosse] Status do Stream: {status}", flush=True)

        audio_data = indata[:, 0]
        scores, embeddings, spectrogram = yamnet_model(audio_data)
        scores_np = scores.numpy()
        
        cough_scores = scores_np[:, COUGH_CLASS_INDEX]
        mean_cough_score = np.mean(cough_scores)

        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        if mean_cough_score > DETECTION_THRESHOLD and cooldown_counter == 0:
            print(f"[Detector de Tosse] TOSSE DETETADA! Pontuação: {mean_cough_score:.2f}")
            
            # **A MÁGICA ACONTECE AQUI**
            # Em vez de incrementar um contador local, chamamos a função que recebemos
            if on_cough_detected_callback:
                on_cough_detected_callback()
            
            # Ativa o cooldown
            sample_rate = sd.query_devices(None, 'input')['default_samplerate']
            cooldown_frames = int(COOLDOWN_SECONDS * (sample_rate / frames))
            cooldown_counter = cooldown_frames

    try:
        sample_rate = 16000
        block_duration = 0.975
        block_size = int(sample_rate * block_duration)

        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            callback=audio_callback,
            dtype='float32',
            blocksize=block_size
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\n[Detector de Tosse] Programa interrompido pelo utilizador.")
    except Exception as e:
        print(f"[Detector de Tosse] Ocorreu um erro: {e}")
        print("[Detector de Tosse] Verifique se tem um microfone ligado e se as permissões estão corretas.")
    finally:
        print("--- Fim da Sessão de Detecção de Tosse ---")