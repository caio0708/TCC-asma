from flask import Blueprint, render_template, session, request, redirect, flash, url_for, jsonify
from datetime import datetime, timedelta
import os
import sqlite3  #  Importado sqlite3 para acesso ao DB
from pathlib import Path  # Para um caminho de arquivo mais robusto
from dotenv import load_dotenv
import csv
from routes.crises import get_user_prediction, get_categorical_mappings
from routes.api import get_air_quality, get_user_location

painel_bp = Blueprint('painel', __name__)

# --- CONSTANTES ---
# O caminho agora aponta para o banco de dados SQLite
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "dados" / "sensores.db"

load_dotenv()

# --- FUNÇÕES DE DADOS ---

def to_float(value):
    try: return float(value)
    except (ValueError, TypeError): return None

def get_historical_data(target_labels):
    """
    Busca dados históricos no banco de dados, encontrando para cada horário-alvo
    o registro de sensor com o tempo mais próximo.
    """
    historical_metrics = {
        "body_temp": [], "ambient_temp": [], "humidity": [], "spo2": []
    }
    target_datetimes = [datetime.strptime(label, "%Y-%m-%d %H:%M") for label in target_labels]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row  # Permite acessar colunas por nome
            cursor = conn.cursor()

            for target_dt in target_datetimes:
                target_timestamp = int(target_dt.timestamp())

                # Query para encontrar a linha com o tempo mais próximo do nosso alvo (hora cheia)
                # Calcula a diferença absoluta em segundos entre cada registro e o alvo,
                # ordena por essa diferença e pega o primeiro (o mais próximo).
                query = """
                SELECT * FROM sensores
                ORDER BY ABS(strftime('%s', Data || ' ' || Hora) - ?)
                LIMIT 1
                """
                cursor.execute(query, (target_timestamp,))
                closest_entry = cursor.fetchone()

                if closest_entry:
                    historical_metrics["body_temp"].append(to_float(closest_entry["temperatura-corporal"]))
                    historical_metrics["ambient_temp"].append(to_float(closest_entry["temperatura-ambiente"]))
                    historical_metrics["humidity"].append(to_float(closest_entry["umidade"]))
                    historical_metrics["spo2"].append(to_float(closest_entry["saturacao"]))
                else:
                    # Se não houver nenhum dado no DB para aquele período, envia None.
                    historical_metrics["body_temp"].append(None)
                    historical_metrics["ambient_temp"].append(None)
                    historical_metrics["humidity"].append(None)
                    historical_metrics["spo2"].append(None)

    except sqlite3.Error as e:
        print(f"Erro ao acessar o banco de dados em get_historical_data: {e}")
        # Preenche com None em caso de falha total no DB
        for key in historical_metrics:
            historical_metrics[key] = [None] * len(target_datetimes)

    return historical_metrics


def _get_dashboard_data(username='Visitante'):
    """Função centralizada para buscar todos os dados do painel."""
    hora_atual = datetime.now().strftime("%H:%M")

    # --- LÓGICA DE PREDIÇÃO (Mantida) ---
    resultado_pred, erro_pred, acuracia_pred, data_pred, hora_pred_api = get_user_prediction(username)
    if erro_pred is None:
        resultado = resultado_pred
        acuracia = acuracia_pred
        hora_predicao = hora_pred_api if isinstance(hora_pred_api, str) and hora_pred_api else hora_atual
    else:
        print(f"Aviso do painel: Não foi possível obter a predição. Motivo: {erro_pred}")
        resultado, acuracia, hora_predicao = 0, acuracia_pred, hora_atual

    def format_string(value, precision=1):
        try: return f"{float(value):.{precision}f}"
        except (ValueError, TypeError): return str(value)

    # --- [ALTERADO] BUSCA DE DADOS MAIS RECENTES DO BANCO DE DADOS ---
    latest_sensor_data = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Query otimizada para pegar a última entrada de sensor
            cursor.execute("SELECT * FROM sensores ORDER BY Data DESC, Hora DESC LIMIT 1")
            latest_row = cursor.fetchone()
            if latest_row:
                latest_sensor_data = dict(latest_row)
    except sqlite3.Error as e:
        print(f"Erro ao buscar dados mais recentes do DB: {e}")


    temperatura_corporal = to_float(latest_sensor_data.get("temperatura-corporal"))
    temperatura_ambiente = to_float(latest_sensor_data.get("temperatura-ambiente"))
    humidity = to_float(latest_sensor_data.get("umidade"))
    nivel_oxi = to_float(latest_sensor_data.get("saturacao"))
    cont_tosse = to_float(latest_sensor_data.get("contagem-tosse"))

    # --- API EXTERNA E DADOS DE AMBIENTE (Mantido) ---
    lat, lon, city = get_user_location()
    api_key = os.getenv('API_WEATHER_KEY')
    aqi, pm2_5, pm10, o3, no2, so2 = get_air_quality(lat, lon, api_key)

    # --- DADOS PARA GRÁFICOS (Gerados a cada requisição) ---
    now = datetime.now().replace(second=0, microsecond=0)
    if now.minute > 0:
        now = (now + timedelta(hours=1)).replace(minute=0)
    else:
        now = now.replace(minute=0)

    labels_for_lookup = [(now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in reversed(range(24))]
    
    # [ALTERADO] Chamada da nova função que consulta o DB
    historical_chart_data = get_historical_data(labels_for_lookup)
    historical_chart_data['labels'] = [datetime.strptime(lbl, "%Y-%m-%d %H:%M").strftime("%H:%M") for lbl in labels_for_lookup]
    
    air_quality = {
        "labels": ["AQI", "PM2.5", "PM10", "O3", "NO2", "SO2"],
        "values": [aqi, pm2_5, pm10, o3, no2, so2]
    }

    daily_summary = [
        {"title": "Temperatura Corporal", "value": format_string(temperatura_corporal, 1), "unit": "°C"},
        {"title": "Temperatura Ambiente", "value": format_string(temperatura_ambiente, 1), "unit": "°C"},
        {"title": "Umidade", "value": format_string(humidity, 0), "unit": "%"},
        {"title": "Saturação O₂", "value": format_string(nivel_oxi, 0), "unit": "%"},
        {"title": "Tosse (dia)", "value": format_string(cont_tosse, 0), "unit": ""}
    ]

    return {
        "daily_summary": daily_summary,
        "air_quality": air_quality,
        "historical_chart_data": historical_chart_data,
        "crisesPrediction": resultado,
        "accuracy": acuracia,
        "hora": hora_predicao,
    }


@painel_bp.route('/')
def painel():
    """Renderiza a página inicial do painel."""
    username = session.get('username', 'Visitante')
    context = _get_dashboard_data(username)
    context['username'] = username
    return render_template('painel.html', **context)


@painel_bp.route('/data')
def data():
    """Fornece os dados do painel em formato JSON para atualizações dinâmicas."""
    username = session.get('username', 'Visitante')
    dashboard_data = _get_dashboard_data(username)
    return jsonify(dashboard_data)

# --- Rotas de Questionário (Mantidas) ---

@painel_bp.route("/questionario")
def questionario():
    return render_template("questionario.html")

@painel_bp.route('/sintomas', methods=['POST'])
def sintomas():
    if 'username' not in session:
        flash('Você precisa estar logado para registrar sintomas.', 'warning')
        return redirect(url_for('configuracoes.configuracoes'))

    data = datetime.now().strftime('%Y-%m-%d')
    hora = datetime.now().strftime('%H:%M')

    # Busca os mapeamentos
    mappings = get_categorical_mappings()

    def get_float(key):
        return to_float(request.form.get(key))

    def get_int(key):
        try:
            return int(request.form.get(key))
        except (ValueError, TypeError):
            return 0
    
    # Converte os valores do formulário para os textos corretos
    sintomas_data = [
        data, hora,
        session.get('age', 0),
        mappings['Gender'].get(session.get('gender', ''), ''), # Mapeia o gênero
        get_float('BMI'),
        mappings['Smoking_Status'].get(get_int('Smoking_Status'), 'Never'),
        get_int('Family_History'), # Mantém como 0 ou 1
        mappings['Allergies'].get(get_int('Allergies'), 'None'),
        mappings['Air_Pollution_Level'].get(get_int('Air_Pollution_Level'), 'Low'),
        mappings['Physical_Activity_Level'].get(get_int('Physical_Activity_Level'), 'Sedentary'),
        mappings['Occupation_Type'].get(get_int('Occupation_Type'), 'Indoor'),
        mappings['Comorbidities'].get(get_int('Comorbidities'), 'None'),
        get_float('Medication_Adherence'), # Mantém como float
        get_int('Peak_Expiratory_Flow'),
        get_int('FeNO_Level'),
        get_int('Number_of_ER_Visits'),
        session.get('username', '')
    ]

    file_exists = os.path.isfile("dados/sintomas.csv")
    with open("dados/sintomas.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                    "Data","Hora","Age","Gender", "BMI","Smoking_Status",
                    "Family_History","Allergies","Air_Pollution_Level","Physical_Activity_Level",
                    "Occupation_Type","Comorbidities","Medication_Adherence",
                    "Peak_Expiratory_Flow","FeNO_Level","Number_of_ER_Visits", "Username"
            ])
        writer.writerow(sintomas_data)

    flash('Sintomas registrados com sucesso!', 'success')
    return redirect(url_for('painel.painel'))