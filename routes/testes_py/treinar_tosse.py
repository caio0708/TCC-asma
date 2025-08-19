import pyaudio
import numpy as np
import librosa
import librosa.display
import os
import glob
import joblib
import time
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parâmetros Globais ---
# Áudio
RATE = 44100
CHUNK = int(RATE / 2) # Captura blocos de 0.5s para maior responsividade
THRESHOLD = 0.02
CAPTURE_TIMEOUT = 10 # Reduzido para agilizar a coleta

# Dados e Modelo
COUGH_DATA_DIR = r'C:\Users\caiot\Downloads\Nova pasta (5)\Cough Detection\data'
NUM_NON_COUGH_SAMPLES = 100
MODEL_FILENAME = "modelo_tosse_aprimorado.pkl"
SCALER_FILENAME = "scaler_tosse_aprimorado.pkl"

# --- Funções de Aumento de Dados (Data Augmentation) ---
def add_noise(audio, noise_factor=0.005):
    """ Adiciona ruído gaussiano ao sinal de áudio. """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_shift(audio, shift_max_ms=50):
    """ Desloca o áudio no tempo por um valor aleatório. """
    shift_samples = int(shift_max_ms * RATE / 1000)
    shift = np.random.randint(-shift_samples, shift_samples)
    return np.roll(audio, shift)

def pitch_shift(audio, sr=RATE, n_steps=2):
    """ Altera o tom (pitch) do áudio. """
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def augment_data(audio, sr=RATE):
    """ Aplica uma ou mais técnicas de aumento de dados aleatoriamente. """
    augmented_audio = audio.copy()
    if random.choice([True, False]):
        augmented_audio = add_noise(augmented_audio)
    if random.choice([True, False]):
        augmented_audio = time_shift(augmented_audio)
    if random.choice([True, False]):
        # Aplica pitch shift com pequena variação para não descaracterizar a tosse
        steps = random.uniform(-1.5, 1.5)
        augmented_audio = pitch_shift(augmented_audio, sr=sr, n_steps=steps)
    return augmented_audio

# --- Funções de Processamento e Extração de Features ---
def preprocess_audio(audio):
    """ Normaliza e aplica pré-ênfase para realçar altas frequências. """
    try:
        # Normaliza o áudio para o intervalo [-1, 1]
        audio = audio / np.max(np.abs(audio))
        # Aplica um filtro de pré-ênfase
        audio = librosa.effects.preemphasis(audio)
        return audio
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None

def extract_features(audio, sr=RATE):
    """ Extrai um conjunto rico de features do áudio. """
    try:
        if len(audio) < 2048: # Garante que o áudio tenha um tamanho mínimo
             return None
             
        audio = preprocess_audio(audio)
        if audio is None:
            return None

        # Features Espectrais (MFCCs e seus deltas)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        mfcc_features = np.concatenate((
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1), np.std(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1), np.std(delta2_mfccs, axis=1)
        ))

        # Contraste Espectral
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_features = np.concatenate((
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1)
        ))

        # Features Temporais
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # Combina todas as features em um único vetor
        features = np.concatenate((mfcc_features, contrast_features, [zcr, rms]))
        return features

    except Exception as e:
        print(f"Erro ao extrair features: {e}")
        return None

# --- Funções de Coleta de Dados ---
def load_cough_features(directory, augment=True):
    """ Carrega áudios de tosse, extrai features e aplica data augmentation. """
    cough_features = []
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    print(f"Encontrados {len(wav_files)} arquivos .wav de tosses.")
    
    for file in wav_files:
        try:
            audio, sr = librosa.load(file, sr=RATE)
            if len(audio) > 0:
                # Feature do áudio original
                features = extract_features(audio, sr)
                if features is not None:
                    cough_features.append(features)
                
                # Aplica augmentation para criar 2 variações sintéticas
                if augment:
                    for _ in range(2):
                        augmented_audio = augment_data(audio, sr)
                        aug_features = extract_features(augmented_audio, sr)
                        if aug_features is not None:
                            cough_features.append(aug_features)

        except Exception as e:
            print(f"Erro ao carregar ou processar {file}: {e}")
            
    print(f"Total de {len(cough_features)} amostras de tosse (com augmentation).")
    return cough_features

def capture_sample(stream, prompt):
    """ Captura uma amostra de áudio do microfone. """
    print(prompt)
    time.sleep(0.5)
    start_time = time.time()
    
    while time.time() - start_time < CAPTURE_TIMEOUT:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            energy = np.sqrt(np.mean(audio**2))
            
            if energy > THRESHOLD:
                print(f"Som detectado com energia: {energy:.4f}. Gravando...")
                # Grava por 1 segundo inteiro para capturar o evento completo
                full_data = data + stream.read(CHUNK, exception_on_overflow=False)
                full_audio = np.frombuffer(full_data, dtype=np.float32)
                print("Amostra capturada!")
                return full_audio
        except Exception as e:
            print(f"Erro na captura de áudio: {e}")
        time.sleep(0.1)
        
    print("Timeout: Nenhum som acima do limiar foi detectado.")
    return None

def main():
    """ Função principal para executar o treinamento do modelo. """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("--- INICIANDO TREINAMENTO OTIMIZADO DO MODELO ---")

    # 1. Carregar e aumentar dados de tosses
    cough_features = load_cough_features(COUGH_DATA_DIR, augment=True)
    if len(cough_features) < 10:
        print("Poucas amostras de tosse encontradas. Abortando.")
        return

    # 2. Coletar amostras de não-tosses (ruído ambiente, fala, etc.)
    non_cough_features = []
    prompts = [
        "Fique em silêncio (ruído de fundo).", "Fale 'olá, como vai você?'.", "Assovie uma melodia.",
        "Bata palmas uma vez.", "Digite algo no teclado.", "Respire fundo.",
        "Mova sua cadeira.", "Diga uma frase longa e contínua.", "Limpe a garganta (som de 'aham').",
        "Estale os dedos perto do microfone.", "Ligue e desligue um ventilador se possível.",
        "Toque um trecho curto de música no celular."
    ]
    print(f"\n--- Fase de Calibração: Capturando {NUM_NON_COUGH_SAMPLES} amostras de 'não-tosses' ---")
    while len(non_cough_features) < NUM_NON_COUGH_SAMPLES:
        idx = len(non_cough_features)
        prompt = f"Amostra {idx+1}/{NUM_NON_COUGH_SAMPLES}: {prompts[idx % len(prompts)]}"
        audio = capture_sample(stream, prompt)
        if audio is not None:
            features = extract_features(audio)
            if features is not None:
                non_cough_features.append(features)
        else:
            print(f"Amostra {idx+1} ignorada.")

    if len(non_cough_features) < 10:
        print("Poucas amostras de não-tosse capturadas. Abortando.")
        return

    # 3. Preparar o dataset
    print("\n--- Preparando dados para o treinamento ---")
    X = np.array(cough_features + non_cough_features)
    y = np.array([1] * len(cough_features) + [0] * len(non_cough_features))
    
    # 4. Dividir em Treino e Teste (75% treino, 25% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Total de amostras: {len(X)}")
    print(f"Amostras de treino: {len(X_train)}, Amostras de teste: {len(X_test)}")

    # 5. Normalizar os dados (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # IMPORTANTE: usar o scaler treinado nos dados de treino

    # 6. Treinar o modelo com GridSearchCV e RandomForest
    print("\n--- Treinando o modelo com validação cruzada ---")
    # Modelo: RandomForest é uma excelente escolha pela sua robustez
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }

    # Busca pelos melhores hiperparâmetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nMelhores hiperparâmetros encontrados: {grid_search.best_params_}")

    # 7. Avaliar o modelo no conjunto de TESTE
    print("\n--- Avaliando o modelo no conjunto de teste (dados não vistos) ---")
    y_pred = best_model.predict(X_test_scaled)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=["Não-Tosse", "Tosse"]))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Não-Tosse", "Tosse"], yticklabels=["Não-Tosse", "Tosse"])
    plt.title('Matriz de Confusão no Conjunto de Teste')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()

    # 8. Salvar o modelo e o scaler
    print(f"\nSalvando o modelo treinado em '{MODEL_FILENAME}'...")
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"Salvando o normalizador (scaler) em '{SCALER_FILENAME}'...")
    joblib.dump(scaler, SCALER_FILENAME)

    print("\nTreinamento concluído com sucesso!")

    # Finalizar PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    main()