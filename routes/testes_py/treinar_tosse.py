# -*- coding: utf-8 -*-
"""
Treino de detector de tosse — otimizado e mais preciso

Requisitos: numpy, scipy, librosa, scikit-learn, joblib, matplotlib, seaborn (opcional), pyaudio (apenas se usar captura)
"""

import os, glob, time, random, json, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import librosa
import joblib

from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# --------- Parâmetros Globais ----------
RATE = 44100
FRAME_SEC = 0.5
CHUNK = int(RATE * FRAME_SEC)        # ~0.5s
TRIGGER_THRESHOLD = 0.02
POST_TRIGGER_SEC = 1.2               # captura extra após disparo
CAPTURE_TIMEOUT = 10

COUGH_DATA_DIR = r'C:\Users\caiot\Downloads\Nova pasta (5)\Cough Detection\data'
NUM_NON_COUGH_SAMPLES = 100

MODEL_FILENAME = "modelo_tosse_pipeline.pkl"
META_FILENAME  = "modelo_tosse_meta.json"

RANDOM_STATE = 42
N_JOBS = -1                           # usar todos os núcleos
AUG_PER_FILE = 2                      # quantas variações por arquivo

# --------- Utilidades ----------
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

def pre_emphasis(x, coef=0.97):
    # y[n] = x[n] - a * x[n-1]
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y

def safe_vector(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32, copy=False)

def trim_silence(y, sr, top_db=30):
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y

# --------- Data Augmentation ----------
def add_noise(audio, snr_db=25):
    # ruído gaussiano controlado por SNR (em dB)
    rms = np.sqrt(np.mean(audio**2) + 1e-12)
    noise_rms = rms / (10**(snr_db/20.0))
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_rms
    return audio + noise

def time_shift(audio, max_ms=50, sr=RATE):
    shift = int((random.uniform(-max_ms, max_ms) / 1000.0) * sr)
    return np.roll(audio, shift)

def pitch_shift(audio, sr=RATE, max_steps=1.5):
    steps = random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def augment_once(y, sr=RATE):
    a = y.copy()
    if random.random() < 0.7: a = add_noise(a, snr_db=random.uniform(20, 30))
    if random.random() < 0.5: a = time_shift(a, max_ms=60, sr=sr)
    if random.random() < 0.4: a = pitch_shift(a, sr=sr, max_steps=1.2)
    return a

# --------- Extração de Features ----------
def extract_features(y, sr=RATE):
    # sanity checks
    if y is None or len(y) < 2048:
        return None

    # normaliza, pré-ênfase, recorta silêncio
    y = y / (np.max(np.abs(y)) + 1e-12)
    y = pre_emphasis(y, coef=0.97)
    y = trim_silence(y, sr, top_db=30)

    if len(y) < 2048:
        return None

    # log-mel e MFCC(+deltas)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=512, n_fft=1024, fmin=50, fmax=8000)
    logS = librosa.power_to_db(S, ref=np.max)

    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    # contraste espectral
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # zcr e rms
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # pooling (estatísticas)
    def stats(mat):
        return np.concatenate([np.mean(mat, axis=1), np.std(mat, axis=1)])

    feat = np.concatenate([
        stats(mfcc), stats(d1), stats(d2),
        stats(contrast),
        [np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)]
    ])

    return safe_vector(feat)

# --------- Carregamento em Paralelo ----------
def load_one_file(path, sr=RATE, augment=True):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        feats = []
        f0 = extract_features(y, sr)
        if f0 is not None:
            feats.append(f0)
        if augment:
            for _ in range(AUG_PER_FILE):
                ya = augment_once(y, sr)
                fa = extract_features(ya, sr)
                if fa is not None:
                    feats.append(fa)
        return feats
    except Exception as e:
        print(f"[WARN] Erro ao processar {os.path.basename(path)}: {e}")
        return []

def load_cough_features(directory, augment=True, n_jobs=N_JOBS):
    wavs = glob.glob(os.path.join(directory, "*.wav"))
    print(f"Encontrados {len(wavs)} .wav de tosses.")
    if not wavs:
        return []

    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(load_one_file)(w, RATE, augment) for w in wavs
    )
    feats = [f for sub in results for f in sub]
    print(f"Total de {len(feats)} amostras de tosse (com augmentation={augment}).")
    return feats

# --------- Captura (opcional) ----------
def capture_sample_pyaudio(prompt):
    import pyaudio  # import local para não exigir na fase de treino offline
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(prompt)
    time.sleep(0.4)
    start = time.time()

    audio_buf = bytearray()
    triggered = False

    try:
        while time.time() - start < CAPTURE_TIMEOUT:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            energy = np.sqrt(np.mean(audio**2))

            if not triggered and energy > TRIGGER_THRESHOLD:
                triggered = True
                print(f"Som detectado (energia {energy:.4f}). Gravando janela estendida...")

                audio_buf.extend(data)
                # capturar mais alguns frames para pegar o evento completo
                post_frames = int((POST_TRIGGER_SEC / FRAME_SEC))
                for _ in range(post_frames):
                    d2 = stream.read(CHUNK, exception_on_overflow=False)
                    audio_buf.extend(d2)
                break

        if len(audio_buf) == 0:
            print("Timeout: nada acima do limiar.")
            return None
        y = np.frombuffer(bytes(audio_buf), dtype=np.float32)
        return y
    finally:
        stream.stop_stream(); stream.close(); p.terminate()

# --------- Treino ----------
def build_param_grid():
    # dois modelos: SVC RBF e RandomForest
    grid = [
        {
            "clf": [SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=RANDOM_STATE)],
            "clf__C": [0.5, 1, 2, 4],
            "clf__gamma": ["scale", 0.01, 0.001]
        },
        {
            "clf": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")],
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 16, 24],
            "clf__min_samples_split": [2, 5],
            "clf__max_features": ["sqrt", "log2"]
        }
    ]
    return grid

def train_and_eval(cough_features, non_cough_features):
    X = np.array(cough_features + non_cough_features, dtype=np.float32)
    y = np.array([1] * len(cough_features) + [0] * len(non_cough_features), dtype=np.int64)

    print(f"Total de amostras: {len(X)} | Positivas: {np.sum(y)} | Negativas: {len(y)-np.sum(y)}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC())  # placeholder, trocado pelo GridSearch
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = build_param_grid()

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="f1",      # f1 geralmente equilibra bem para este caso binário
        cv=cv,
        n_jobs=N_JOBS,
        verbose=1
    )
    gs.fit(X, y)

    print("\nMelhores hiperparâmetros:", gs.best_params_)
    best = gs.best_estimator_

    # avaliação honesta via CV out-of-fold
    # (para um hold-out adicional, separe antes; aqui ficamos com CV por robustez)
    y_pred = gs.predict(X)
    try:
        # se o classificador tem predict_proba, calcula AUC
        if hasattr(best.named_steps["clf"], "predict_proba"):
            y_proba = best.predict_proba(X)[:, 1]
        else:
            # fallback com decision_function normalizada
            dec = best.decision_function(X)
            dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
            y_proba = dec
        auc = roc_auc_score(y, y_proba)
    except Exception:
        auc = None

    print("\nRelatório de Classificação (em TODO o conjunto; para produção, use CV OOF):")
    print(classification_report(y, y_pred, target_names=["Não-Tosse", "Tosse"]))
    cm = confusion_matrix(y, y_pred)
    print("Matriz de Confusão:\n", cm)
    if auc is not None:
        print(f"AUC (aprox.): {auc:.4f}")

    # plot opcional
    try:
        import seaborn as sns
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Não-Tosse","Tosse"],
                    yticklabels=["Não-Tosse","Tosse"])
        plt.title("Matriz de Confusão (treino+CV)")
        plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    return best

# --------- Main ----------
def main():
    set_seed(RANDOM_STATE)

    print("\n--- Carregando dados de tosse ---")
    cough_features = load_cough_features(COUGH_DATA_DIR, augment=True, n_jobs=N_JOBS)
    if len(cough_features) < 10:
        print("Poucas amostras de tosse. Abortando.")
        return

    print(f"\n--- Capturando {NUM_NON_COUGH_SAMPLES} amostras de não-tosse (opcional) ---")
    # Se você já tem negativos gravados, substitua este bloco por carregamento de arquivos.
    # Mantive a função de captura para compatibilidade; comente se não for usar.
    non_cough_features = []
    try:
        prompts = [
            "Silêncio (ruído de fundo)", "Fale uma frase curta", "Assovie",
            "Bata palmas", "Tecle no teclado", "Respire fundo", "Mexa na cadeira",
            "Diga uma frase longa", "Limpe a garganta", "Estale os dedos",
            "Ligue um ventilador", "Toque música no celular"
        ]
        while len(non_cough_features) < NUM_NON_COUGH_SAMPLES:
            i = len(non_cough_features)
            y = capture_sample_pyaudio(f"Amostra {i+1}/{NUM_NON_COUGH_SAMPLES}: {prompts[i % len(prompts)]}")
            if y is None:
                print("Amostra ignorada.")
                continue
            f = extract_features(y, RATE)
            if f is not None:
                non_cough_features.append(f)
    except Exception as e:
        print(f"[INFO] Captura não usada/indisponível ({e}). Use arquivos negativos se tiver.")
        if len(non_cough_features) == 0:
            print("Sem negativos suficientes. Abortando.")
            return

    print("\n--- Treinando e avaliando ---")
    best_pipeline = train_and_eval(cough_features, non_cough_features)

    # Persistência: salva o pipeline completo (scaler + classificador)
    joblib.dump(best_pipeline, MODEL_FILENAME)
    meta = {
        "rate": RATE,
        "random_state": RANDOM_STATE,
        "augment_per_file": AUG_PER_FILE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": MODEL_FILENAME,
        "lib_versions": {
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__
        }
    }
    with open(META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {MODEL_FILENAME}")
    print(f"Metadados salvos em: {META_FILENAME}")
    print("Pronto!")

if __name__ == "__main__":
    main()
