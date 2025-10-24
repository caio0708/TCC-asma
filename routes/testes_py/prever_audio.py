##############################################################################
# 1) IMPORTS E CONFIGURAÇÃO INICIAL
##############################################################################
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CAMINHOS DOS ARQUIVOS ---
MODEL_ARTIFACTS_DIR = r'E:\Dev\TCC-asma\ia\model_artifacts'
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'audio_asthma_detection_model.keras')
ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder.joblib')
NEW_AUDIO_PATH = r'E:\Dev\TCC-asma\ia\uploads\audio\gravacao_respiracao.wav'

##############################################################################
# 2) CARREGAMENTO DO MODELO E DO ENCODER
##############################################################################
model = None
encoder = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo '{os.path.basename(MODEL_PATH)}' carregado com sucesso.")
    encoder = joblib.load(ENCODER_PATH)
    print(f"Encoder '{os.path.basename(ENCODER_PATH)}' carregado com sucesso.")

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique os caminhos. Detalhes: {e}")
except Exception as e:
    print(f"Ocorreu um erro ao carregar o modelo ou o encoder: {e}")

##############################################################################
# 3) PROCESSO DE PREDIÇÃO (se o modelo e o encoder foram carregados)
##############################################################################
if model and encoder:
    try:
        # --- ETAPA A: Carregar o novo áudio ---
        print("\n--- Iniciando Predição ---")
        y_new, sr_new = librosa.load(NEW_AUDIO_PATH, sr=None)
        print(f"Áudio '{os.path.basename(NEW_AUDIO_PATH)}' carregado com sucesso.")


        ##############################################################################
        # Teste com um novo áudio: Extrair features
        ##############################################################################
        if y_new is not None and sr_new is not None:
            try:
                # Add error handling for short audio files
                if len(y_new) < 2048: # Minimum required length for default n_fft
                    print(f"Skipping feature extraction for the new audio sample (length: {len(y_new)}) because it's too short for default n_fft.")
                    X_new = None # Ensure X_new is None if skipping
                else:
                    mfcc_new = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
                    X_new = np.mean(mfcc_new, axis=1)
                    print(f"Features extraídas com sucesso. Forma: {X_new.shape}")
            except Exception as e:
                print(f"Erro ao extrair features: {e}")
                X_new = None # Ensure X_new is None if extraction fails
        else:
            print("Não foi possível extrair features. O áudio não foi carregado corretamente.")
            X_new = None # Ensure X_new is None if audio was not loaded


        ##############################################################################
        # Teste com um novo áudio: Pré-processar features (Reshape para LSTM)
        ##############################################################################
        if X_new is not None:
            # O modelo LSTM espera entrada 3D (batch_size, timesteps, channels).
            # Nossas features são atualmente 1D (num_features,).
            # Precisamos remodelá-las para (1, num_features, 1) para um único sample.
            try:
                X_new_processed = X_new.reshape(1, X_new.shape[0], 1)
                print(f"Features pré-processadas com sucesso. Nova forma: {X_new_processed.shape}")
            except Exception as e:
                print(f"Erro ao pré-processar features: {e}")
                X_new_processed = None # Ensure X_new_processed is None if reshaping fails
        else:
            print("Não foi possível pré-processar features. Features não foram extraídas corretamente.")
            X_new_processed = None # Ensure X_new_processed is None if features were not extracted


        ##############################################################################
        # Teste com um novo áudio: Fazer a predição
        ##############################################################################
        if X_new_processed is not None:
            try:
                # Use o modelo treinado para prever as probabilidades das classes
                y_new_pred_proba = model.predict(X_new_processed)
                print(f"Predição de probabilidades bruta: {y_new_pred_proba}")

                # Obtenha a classe prevista (índice com a maior probabilidade)
                y_new_pred_encoded = np.argmax(y_new_pred_proba, axis=1)
                print(f"Índice da classe prevista (encoded): {y_new_pred_encoded}")

            except Exception as e:
                print(f"Erro ao fazer a predição com o modelo: {e}")
                y_new_pred_encoded = None # Ensure this is None if prediction fails
        else:
            print("Não foi possível fazer a predição. Features pré-processadas não disponíveis.")
            y_new_pred_encoded = None # Ensure this is None if features are not available


        ##############################################################################
        # Teste com um novo áudio: Interpretar a predição e exibir o resultado
        ##############################################################################
        if y_new_pred_encoded is not None:
            try:
                # Use o encoder para obter o nome da classe original
                predicted_class = encoder.inverse_transform(y_new_pred_encoded)
                print(f"\nA classe prevista para o áudio é: {predicted_class[0]}") # predicted_class is an array
            except Exception as e:
                print(f"Erro ao interpretar a predição: {e}")
        else:
            print("Não foi possível interpretar a predição. Predição não disponível.")

    except Exception as e:
        print(f"\nOcorreu um erro durante o processo de predição: {e}")
else:
    print("\nA predição não pode ser executada porque o modelo ou o encoder não foram carregados corretamente.")