import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def guia_respiracao(ciclos=3):
    print("\n🌬️  Guia de Respiração")
    for i in range(ciclos):
        print(f"\nCiclo {i+1}:")
        print("→ Inspire profundamente pelo nariz...")
        time.sleep(3)
        print("→ Pausa...")
        time.sleep(1)
        print("→ Expire lentamente pela boca...")
        time.sleep(3)
        print("→ Pausa...\n")
        time.sleep(1)

def gravar_audio(filename="respiracao.wav", duracao=10, fs=44100):
    print("\n🎙️  Gravando áudio...")
    audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ Áudio salvo como: {filename}")

def gerar_spectrograma(audio_path):
    print("\n📊 Gerando espectrograma...")

    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma de Frequência da Respiração")
    plt.tight_layout()
    
    img_path = os.path.splitext(audio_path)[0] + "_spectrograma.png"
    plt.savefig(img_path)
    plt.show()

    print(f"✅ Espectrograma salvo como: {img_path}")

# ----------- EXECUÇÃO -----------
print("Mini-App de Gravação e Análise de Respiração\n")
input("Pressione ENTER para começar o guia de respiração...")

guia_respiracao(ciclos=2)  # Ajuste quantos ciclos quiser

input("\nPressione ENTER para iniciar a gravação da sua respiração por 10 segundos...")

gravar_audio("respiracao.wav", duracao=10)

gerar_spectrograma("respiracao.wav")

