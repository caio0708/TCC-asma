import pyaudio
import numpy as np
import librosa
import librosa.display
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import time
import os
import glob
import joblib

# Parâmetros de áudio
RATE = 44100
CHUNK = int(RATE)
THRESHOLD = 0.02
NUM_NON_COUGH_SAMPLES = 100  # Aumentado para maior diversidade
COUGH_DATA_DIR = r'C:\Users\caiot\Downloads\Nova pasta (5)\Cough Detection\data'
CAPTURE_TIMEOUT = 15
MODEL_FILENAME = "modelo_tosse.pkl"
SCALER_FILENAME = "scaler_tosse.pkl"

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

# Função para extrair features MFCC, temporais e de waveform
def extract_features(audio, sr=RATE):
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
        stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        # Combina todas as features
        features = np.concatenate([mfcc_mean, mfcc_std, [rms, zcr, ste, spectral_centroid, spectral_bandwidth]])
        return features
    except Exception as e:
        print(f"Erro ao extrair features: {e}")
        return None

# Função para carregar features de arquivos .wav de tosses
def load_cough_features(directory):
    cough_features = []
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    print(f"Encontrados {len(wav_files)} arquivos .wav de tosses para treinamento.")
    for file in wav_files:
        try:
            audio, sr = librosa.load(file, sr=RATE)
            if len(audio) > 0:
                features = extract_features(audio, sr)
                if features is not None:
                    cough_features.append(features)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")
    return cough_features

# Função para capturar uma amostra de áudio com timeout
def capture_sample(stream, prompt):
    print(prompt)
    time.sleep(1)
    start_time = time.time()
    while time.time() - start_time < CAPTURE_TIMEOUT:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            energy = np.sqrt(np.mean(audio**2))
            print(f"Energia detectada: {energy:.6f} (limiar: {THRESHOLD})")
            if energy > THRESHOLD:
                print("Amostra capturada!")
                return audio
        except Exception as e:
            print(f"Erro na captura de áudio: {e}")
        time.sleep(0.1)
    print("Timeout: Nenhuma amostra com energia suficiente detectada.")
    return None

# Inicializa o PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("--- INICIANDO TREINAMENTO DO MODELO ---")

# Carrega features de tosses dos arquivos .wav
cough_features = load_cough_features(COUGH_DATA_DIR)
if len(cough_features) == 0:
    print("Nenhum arquivo de tosse válido encontrado. Abortando.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    exit()

# Fase de calibração: Coletar amostras de não-tosses
non_cough_features = []
prompts = [
    "Fique em silêncio para capturar ruído de fundo.",
    "Fale algo (ex.: 'teste, teste') para capturar fala normal.",
    "Assovie por alguns segundos.",
    "Faça um ruído leve, como bater palmas ou estalar os dedos.",
    "Mantenha o ambiente natural (ex.: ruídos de fundo, como ventilador ou teclado).",
    "Fale uma frase longa para capturar fala contínua.",
    "Assovie novamente com um tom diferente.",
    "Faça um som de respiração profunda.",
    "Bata levemente na mesa ou faça outro ruído curto.",
    "Fique em silêncio novamente para capturar variação.",
    "Toque um som ambiente (ex.: música ao fundo).",
    "Faça um som de movimento (ex.: arrastar uma cadeira).",
    "Fale em tom alto, como se estivesse gritando.",
    "Faça um som de impacto (ex.: bater em uma porta)."
]
print("\nFase de calibração: Capturaremos 100 amostras de não-tosses. Siga as instruções.")
for i in range(NUM_NON_COUGH_SAMPLES):
    audio = capture_sample(stream, f"Capturando amostra não-tosse {i+1}/{NUM_NON_COUGH_SAMPLES}: {prompts[i % len(prompts)]}")
    if audio is not None:
        features = extract_features(audio)
        if features is not None:
            non_cough_features.append(features)
    else:
        print(f"Amostra {i+1} ignorada devido a timeout ou erro.")

if len(non_cough_features) < 10:
    print("Menos de 10 amostras de não-tosse válidas capturadas. Abortando.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    exit()

# Balanceamento de classes
min_samples = min(len(cough_features), len(non_cough_features))
cough_features = cough_features[:min_samples]
non_cough_features = non_cough_features[:min_samples]
print(f"Balanceamento aplicado: {min_samples} amostras de tosses e {min_samples} de não-tosses.")

# Preparar dados para treinamento
print("\nPreparando dados e treinando o modelo...")
X = np.vstack((cough_features, non_cough_features))
y = np.array([1] * len(cough_features) + [0] * len(non_cough_features))

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ajuste de hiperparâmetros com GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
model = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, n_jobs=-1)
model.fit(X_scaled, y)

# Exibe os melhores hiperparâmetros
print(f"Melhores hiperparâmetros: {model.best_params_}")

# Validação cruzada e métricas detalhadas
scores = cross_val_score(model.best_estimator_, X_scaled, y, cv=5)
print(f"Acurácia média (validação cruzada): {np.mean(scores):.2f} (±{np.std(scores):.2f})")

# Relatório detalhado
y_pred = model.predict(X_scaled)
print("\nRelatório de classificação no conjunto de treinamento:")
print(classification_report(y, y_pred, target_names=["Não-Tosse", "Tosse"]))

# Salvamento do modelo e scaler
print(f"Salvando o modelo em '{MODEL_FILENAME}'...")
joblib.dump(model.best_estimator_, MODEL_FILENAME)
print(f"Salvando o scaler em '{SCALER_FILENAME}'...")
joblib.dump(scaler, SCALER_FILENAME)

print("\nModelo e scaler salvos com sucesso! Você já pode rodar o script 'detectar_tosse.py'.")

# Fecha o stream
stream.stop_stream()
stream.close()
p.terminate()