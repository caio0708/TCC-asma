import pyaudio
import numpy as np
import librosa
import time
import os
import joblib

# Parâmetros de áudio
RATE = 44100
CHUNK = int(RATE)  # Processa 1 segundo de áudio por vez

# Função para pré-processar áudio (normalização e remoção de ruído)
def preprocess_audio(audio, sr=RATE):
    try:
        # Normaliza o áudio
        audio = audio / np.max(np.abs(audio))
        # Aplica um filtro passa-alta para reduzir ruídos de baixa frequência
        audio = librosa.effects.preemphasis(audio)
        return audio
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None

# Função para extrair features (idêntica à usada no treinamento)
def extract_features(audio, sr=RATE):
    """
    Extrai as features (MFCC, RMS, ZCR, STE, spectral centroid, spectral bandwidth) de um trecho de áudio.
    Esta função deve ser idêntica à usada no treinamento.
    """
    try:
        # Pré-processa o áudio
        audio = preprocess_audio(audio, sr)
        if audio is None:
            return None

        # MFCC
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = mfcc_features.mean(axis=1)
        mfcc_std = mfcc_features.std(axis=1)

        # RMS (energia)
        rms = np.mean(librosa.feature.rms(y=audio))

        # ZCR (taxa de cruzamento por zero)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # Short-Time Energy (STE) em janelas
        frame_length = 2048
        hop_length = 512
        ste = np.mean(librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length))

        # STFT para análise espectral
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        # Combina todas as features
        features = np.concatenate([mfcc_mean, mfcc_std, [rms, zcr, ste, spectral_centroid, spectral_bandwidth]])
        return features
    except Exception as e:
        print(f"Erro ao extrair features: {e}")
        return None

# Função para calibrar o limiar de energia
def calibrate_energy_threshold(stream, duration=5):
    print("Calibrando limiar de energia...")
    energies = []
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            energy = np.sqrt(np.mean(audio**2))
            energies.append(energy)
            time.sleep(0.01)
        except Exception:
            continue
    if energies:
        return np.mean(energies) + 2 * np.std(energies)  # Limiar dinâmico
    return 0.01  # Valor padrão

# Função para detecção de tosses ao vivo
def live_cough_counter(model, scaler, update_callback, energy_threshold=0.01, prob_threshold=0.60):
    """
    Inicia a detecção de tosse ao vivo usando um modelo e scaler pré-treinados.

    Args:
        model: O modelo de classificação (SVC) treinado e carregado.
        scaler: O StandardScaler treinado e carregado.
        update_callback: Uma função que será chamada sem argumentos toda vez que uma tosse for detectada.
        energy_threshold (float): O limiar de energia de áudio para acionar a análise.
        prob_threshold (float): O limiar de probabilidade para classificar um som como tosse.
    """
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        # Calibra o limiar de energia
        energy_threshold = calibrate_energy_threshold(stream)
        print(f"Limiar de energia calibrado: {energy_threshold:.6f}")

        print("\nOuvindo tosses ao vivo... Pressione Ctrl+C para parar.")
        
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            energy = np.sqrt(np.mean(audio**2))

            if energy > energy_threshold:
                features = extract_features(audio)
                if features is not None:
                    # Usa o scaler carregado para transformar as novas features
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # Usa o modelo carregado para fazer a predição
                    prediction = model.predict(features_scaled)
                    prob = model.predict_proba(features_scaled)[0][1]

                    print(f"Energia: {energy:.6f}, Prob: {prob:.2f}")

                    if prediction[0] == 1 and prob > prob_threshold:
                        print(f"Tosse detectada (prob: {prob:.2f})!")
                        update_callback()
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nDetecção interrompida pelo usuário.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        # Garante que os recursos de áudio sejam liberados
        print("Fechando o stream de áudio.")
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()