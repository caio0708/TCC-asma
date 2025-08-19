from flask import Blueprint, render_template, session, request, redirect, flash, url_for, jsonify # Adicionar jsonify
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import csv
from routes.crises import get_user_prediction

from routes.api import get_air_quality, get_user_location

painel_bp = Blueprint('painel', __name__)

# --- CONSTANTES ---
SENSOR_DATA_CSV = 'dados/sensores.csv'
load_dotenv()

# --- FUNÇÕES DE DADOS ---

def to_float(value):
    try: return float(value)
    except (ValueError, TypeError): return None

def read_sensor_data_from_csv(file_path):
    """Lê todos os dados históricos de sensores do arquivo CSV."""
    if not os.path.isfile(file_path):
        print(f"AVISO: Arquivo de dados não encontrado em '{file_path}'")
        return []
    try:
        with open(file_path, mode='r', encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV '{file_path}': {e}")
        return []

def get_historical_data(all_data, target_labels):
    """Alinha dados históricos com horários-alvo."""
    historical_metrics = {
        "body_temp": [], "ambient_temp": [], "humidity": [], "spo2": []
    }
    
    processed_data = []
    for entry in all_data:
        try:
            hora_str = entry.get('Hora', '')[:5]  # Corta segundos
            entry_dt = datetime.strptime(f"{entry.get('Data', '')} {hora_str}", "%Y-%m-%d %H:%M")
            entry['datetime'] = entry_dt
            processed_data.append(entry)
        except (ValueError, TypeError):
            continue
    
    sorted_data = sorted(processed_data, key=lambda x: x['datetime'])
    target_datetimes = [datetime.strptime(label, "%Y-%m-%d %H:%M") for label in target_labels]
    
    def to_float(value):
        try: return float(value)
        except (ValueError, TypeError): return None

    data_idx = 0
    for target_dt in target_datetimes:
        last_valid_entry = None
        temp_idx = data_idx
        while temp_idx < len(sorted_data) and sorted_data[temp_idx]['datetime'] <= target_dt:
            last_valid_entry = sorted_data[temp_idx]
            # Otimização: Não precisamos recomeçar a busca do início para o próximo target_dt
            if sorted_data[temp_idx]['datetime'] > (target_dt - timedelta(hours=1)):
                data_idx = temp_idx
            temp_idx += 1
        
        if last_valid_entry:
            historical_metrics["body_temp"].append(to_float(last_valid_entry.get("temperatura-corporal")))
            historical_metrics["ambient_temp"].append(to_float(last_valid_entry.get("temperatura-ambiente")))
            historical_metrics["humidity"].append(to_float(last_valid_entry.get("umidade")))
            historical_metrics["spo2"].append(to_float(last_valid_entry.get("saturacao")))
        else:
            # Enviar None (null em JSON) é o correto para o Chart.js
            historical_metrics["body_temp"].append(None)
            historical_metrics["ambient_temp"].append(None)
            historical_metrics["humidity"].append(None)
            historical_metrics["spo2"].append(None)
            
    return historical_metrics

def _get_dashboard_data(username='Visitante'):
    """Função centralizada para buscar todos os dados do painel."""
    hora_atual = datetime.now().strftime("%H:%M")

    # --- CORREÇÃO APLICADA AQUI ---
    # Chamamos a função de predição e obtemos todos os seus retornos
    resultado_pred, erro_pred, acuracia_pred, data_pred, hora_pred_api = get_user_prediction(username)

    # Verificamos se houve um erro. Se não houve, usamos os valores retornados.
    # Se houve um erro, usamos valores padrão e mostramos o erro no console.
    if erro_pred is None:
        resultado = resultado_pred
        acuracia = acuracia_pred
        hora_predicao = hora_pred_api if isinstance(hora_pred_api, str) and hora_pred_api else hora_atual
    else:
        print(f"Aviso do painel: Não foi possível obter a predição. Motivo: {erro_pred}")
        resultado = 0  # Valor padrão para 'sem crise'
        acuracia = acuracia_pred # Ainda usamos a acurácia do modelo, que é carregada independentemente
        hora_predicao = hora_atual

    def to_float(value, default=0.0):
        try: return float(value)
        except (ValueError, TypeError): return default

    def format_string(value, precision=1):
        try: return f"{float(value):.{precision}f}"
        except (ValueError, TypeError): return str(value)

    all_sensor_data = read_sensor_data_from_csv(SENSOR_DATA_CSV)
    
    latest_sensor_data = {}
    if all_sensor_data:
        try:
            all_sensor_data.sort(key=lambda x: datetime.strptime(f"{x.get('Data', '')} {x.get('Hora', '')}", "%Y-%m-%d %H:%M"), reverse=True)
            latest_sensor_data = all_sensor_data[0]
        except (ValueError, TypeError, IndexError):
            latest_sensor_data = all_sensor_data[-1] if all_sensor_data else {}

    temperatura_corporal = to_float(latest_sensor_data.get("temperatura-corporal"))
    temperatura_ambiente = to_float(latest_sensor_data.get("temperatura-ambiente"))
    humidity = to_float(latest_sensor_data.get("umidade"))
    nivel_oxi = to_float(latest_sensor_data.get("saturacao"))
    cont_tosse = to_float(latest_sensor_data.get("contagem-tosse"))

    lat, lon, city = get_user_location()
    api_key = os.getenv('API_WEATHER_KEY')
    aqi, pm2_5, pm10 = get_air_quality(lat, lon, api_key)

    # 1. Valores padrão são definidos aqui
    resultado, acuracia, hora_predicao = 0, 0.0, hora_atual 
    try:
        # 2. A função de predição é chamada
        resultado_pred, _, acuracia_pred, _, hora_pred_api = get_user_prediction(username)
        
        # 3. Os valores são atualizados AQUI DENTRO
        resultado = resultado_pred or 0
        acuracia = float(acuracia_pred) if acuracia_pred else 0.0
        hora_predicao = hora_pred_api if isinstance(hora_pred_api, str) and hora_pred_api else hora_atual

    # 4. SE OCORRER QUALQUER ERRO, este 'except' é ativado e não faz NADA
    except (TypeError, ValueError, Exception):
        pass # O erro é silenciosamente ignorado

    # --- DADOS PARA GRÁFICOS (Gerados a cada requisição) ---
    now = datetime.now().replace(second=0, microsecond=0)
    # Garante que o último rótulo seja a hora atual (arredondada para cima)
    if now.minute > 0:
        now += timedelta(hours=1)
    now = now.replace(minute=0)

    labels_for_lookup = [(now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in reversed(range(24))]
    historical_chart_data = get_historical_data(all_sensor_data, labels_for_lookup)
    historical_chart_data['labels'] = [datetime.strptime(lbl, "%Y-%m-%d %H:%M").strftime("%H:%M") for lbl in labels_for_lookup]
    
    # [CORREÇÃO] Removida a lógica que substituía None pelo valor mais recente.
    # Deixar None é o correto para o Chart.js renderizar os gráficos com falhas nos dados.

    air_quality = {
        "labels": ["AQI", "PM2.5", "PM10"],
        "values": [aqi, pm2_5, pm10]
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
    # Obtém todos os dados da função auxiliar
    context = _get_dashboard_data(username)
    context['username'] = username
    
    return render_template('painel.html', **context)

# [NOVA ROTA] - Endpoint para o JavaScript buscar atualizações
@painel_bp.route('/data')
def data():
    """Fornece os dados do painel em formato JSON para atualizações dinâmicas."""
    username = session.get('username', 'Visitante')
    dashboard_data = _get_dashboard_data(username)
    return jsonify(dashboard_data)

@painel_bp.route("/questionario")
def questionario():
    return render_template("questionario.html")

@painel_bp.route("/salvar_sintomas", methods=["POST"])
def salvar_sintomas():
    if request.method == 'POST':
        if 'username' not in session:
            flash('Faça login primeiro.', 'warning')
            return redirect(url_for('configuracoes.configuracoes'))

        # Coleta dos dados do formulário
        data = request.form.get('data') or datetime.now().strftime('%Y-%m-%d')
        hora = request.form.get('hora') or datetime.now().strftime('%H:%M')

        def get_int(name):
            val = request.form.get(name)
            try: return int(val)
            except: return 0

        def get_float(name):
            val = request.form.get(name)
            try: return float(val)
            except: return 0.0

        def get_str(name):
            return request.form.get(name, '').strip()

        # Features essenciais do modelo
        sintomas_data = [
            data, hora,
            session.get('age', 0),
            session.get('gender', ''),
            get_float('BMI'),
            get_int('Smoking_Status'),
            get_int('Family_History'),
            get_int('Allergies'),
            get_int('Air_Pollution_Level'),
            get_int('Physical_Activity_Level'),
            get_int('Occupation_Type'),
            get_int('Comorbidities'),
            get_int('Medication_Adherence'),
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
                        "Occupation_Type","Comorbidities","Medication_Adherence","Peak_Expiratory_Flow",
                        "FeNO_Level","Number_of_ER_Visits","Username"
                    ])
            writer.writerow(sintomas_data)

        return redirect("/")