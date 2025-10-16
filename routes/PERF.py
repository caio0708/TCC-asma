# routes/PERF.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- Configuração de Caminhos ---
# Define os caminhos baseados na localização deste arquivo (routes/PERF.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "PEFR_predictor_RandomForest.joblib")
USUARIOS_CSV_PATH = os.path.join(BASE_DIR, "dados", "usuarios.csv")
PEFR_DATA_CSV_PATH = os.path.join(BASE_DIR, "dados", "PEFR_Data_Set.csv")

def train_and_save_model():
    """
    Função para treinar o modelo de RandomForest com base no dataset de PEFR
    e salvá-lo para uso futuro. Esta função só precisa ser executada uma vez
    ou quando o modelo precisar ser atualizado.
    """
    print("Iniciando treinamento do modelo de predição de PERF...")
    try:
        data = pd.read_csv(PEFR_DATA_CSV_PATH)
        
        # Remove colunas que não serão usadas como features
        X = data.drop(columns=['PEFR'])
        y = data['PEFR']
        
        # O RandomForest se mostrou o melhor modelo na sua análise inicial
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Garante que o diretório para salvar o modelo exista
        os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Modelo de PERF treinado e salvo em: {MODEL_PATH}")
        
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo de dados '{PEFR_DATA_CSV_PATH}' não encontrado.")
    except Exception as e:
        print(f"❌ ERRO inesperado durante o treinamento do modelo: {e}")

def predict_perf(sensor_data):
    """
    Prevê o valor de PERF com base nos dados do último usuário cadastrado
    e nos dados de sensores recebidos em tempo real.

    Args:
        sensor_data (dict): Dicionário contendo os valores atuais dos sensores.

    Returns:
        dict: Um dicionário com o PERF previsto, PERF de referência,
              percentual e a zona de risco (SAFE, MODERATE, RISK).
    """
    try:
        # Se o modelo não existir, treina e salva automaticamente
        if not os.path.exists(MODEL_PATH):
            print("⚠️ Modelo de PERF não encontrado. Iniciando treinamento...")
            train_and_save_model()

        model = joblib.load(MODEL_PATH)
        
        # 1. Carrega dados do último usuário
        usuarios_pessoal = pd.read_csv(USUARIOS_CSV_PATH)
        ultimo_usuario = usuarios_pessoal.iloc[-1]

        genero = 1 if ultimo_usuario["Gender"].strip().lower() == "masculino" else 0
        idade = int(ultimo_usuario["Age"])
        altura = float(ultimo_usuario["Altura"])

        # 2. Obtém dados dos sensores a partir do dicionário passado como argumento
        p = float(sensor_data.get("temperatura-ambiente", 0))
        q = float(sensor_data.get("umidade", 0))
        r = float(sensor_data.get("qualidade-ar-pm25", 0))
        s = float(sensor_data.get("qualidade-ar-pm10", 0))

        # 3. Faz a predição
        prediction = model.predict([[idade, altura, genero, p, q, r, s]])
        predicted_pefr = prediction[0]

        # 4. Calcula o PERF de referência
        pefr_ref = 0
        if idade < 18:
            pefr_ref = ((altura - 100) * 5) + 100
        else:
            if genero == 1:  # Masculino
                pefr_ref = (((5.48 * (altura / 100)) + 1.58) - (0.041 * idade)) * 60
            else:  # Feminino
                # CORREÇÃO: O código original usava vírgula (2,24) criando uma tupla. Corrigido para ponto.
                pefr_ref = ((((altura / 100) * 3.72) + 2.24) - (idade * 0.03)) * 60
        
        # Evita divisão por zero se pefr_ref for 0
        perpefr = (predicted_pefr / pefr_ref) * 100 if pefr_ref > 0 else 0

        # 5. Define a zona de risco
        zone = "RISK"
        if perpefr >= 80:
            zone = "SAFE"
        elif perpefr >= 50:
            zone = "MODERATE"

        return {
            "predicted_pefr": round(predicted_pefr, 2),
            "reference_pefr": round(pefr_ref, 2),
            "percentage": round(perpefr, 2),
            "zone": zone
        }
    except FileNotFoundError as e:
        print(f"❌ ERRO em predict_perf: Arquivo não encontrado - {e}")
        return {"error": f"Arquivo de dados não encontrado: {e.filename}"}
    except Exception as e:
        print(f"❌ ERRO inesperado em predict_perf: {e}")
        return {"error": f"Ocorreu um erro ao calcular o PERF: {str(e)}"}

# Bloco para permitir treinar o modelo manualmente executando "python routes/PERF.py" no terminal
if __name__ == '__main__':
    train_and_save_model()