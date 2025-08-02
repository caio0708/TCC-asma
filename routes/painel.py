# painel.py - CORRIGIDO PARA ENVIAR DADOS NUMÉRICOS PARA O GRÁFICO

from flask import Blueprint, render_template, session
from datetime import datetime, timedelta
import os
import csv
from routes.crises import get_user_prediction

painel_bp = Blueprint('painel', __name__)

# --- CONSTANTES ---
SENSOR_DATA_CSV = 'dados/sensores.csv'

# --- FUNÇÕES DE DADOS ---

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
            entry_dt = datetime.strptime(f"{entry.get('Data', '')} {entry.get('Hora', '')}", "%Y-%m-%d %H:%M")
            entry['datetime'] = entry_dt
            processed_data.append(entry)
        except (ValueError, TypeError):
            continue
    
    sorted_data = sorted(processed_data, key=lambda x: x['datetime'])
    target_datetimes = [datetime.strptime(label, "%Y-%m-%d %H:%M") for label in target_labels]
    
    def to_float(value):
        """Converte um valor para float de forma segura, retornando None se inválido."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    for target_dt in target_datetimes:
        # Encontrar a entrada mais recente antes ou no horário-alvo
        valid_entries = [entry for entry in sorted_data if entry['datetime'] <= target_dt]
        last_entry = valid_entries[-1] if valid_entries else None
        
        if last_entry:
            historical_metrics["body_temp"].append(to_float(last_entry.get("temperatura-corporal")))
            historical_metrics["ambient_temp"].append(to_float(last_entry.get("temperatura-ambiente")))
            historical_metrics["humidity"].append(to_float(last_entry.get("umidade")))
            historical_metrics["spo2"].append(to_float(last_entry.get("saturacao")))
        else:
            # Se não houver dados, adicionar None para manter o alinhamento
            historical_metrics["body_temp"].append(None)
            historical_metrics["ambient_temp"].append(None)
            historical_metrics["humidity"].append(None)
            historical_metrics["spo2"].append(None)
            
    return historical_metrics

@painel_bp.route('/')
def painel():
    username = session.get('username', 'Visitante')
    hora_atual = datetime.now().strftime("%H:%M")

    # --- FUNÇÕES AUXILIARES ---
    def to_float(value, default=0.0):
        """Converte um valor para float de forma segura."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def format_string(value, precision=1):
        """Formata um valor numérico como string para exibição."""
        try:
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return str(value)

    # --- DADOS DE SENSORES (Fonte única: CSV) ---
    all_sensor_data = read_sensor_data_from_csv(SENSOR_DATA_CSV)
    
    latest_sensor_data = {}
    if all_sensor_data:
        try:
            all_sensor_data.sort(key=lambda x: datetime.strptime(f"{x.get('Data', '')} {x.get('Hora', '')}", "%Y-%m-%d %H:%M"), reverse=True)
            latest_sensor_data = all_sensor_data[0]
        except (ValueError, TypeError, IndexError):
            print("AVISO: Não foi possível obter o dado mais recente do CSV.")
            if all_sensor_data:
                latest_sensor_data = all_sensor_data[-1]

    # Extrai os valores usando os novos nomes de coluna
    temperatura_corporal = to_float(latest_sensor_data.get("temperatura-corporal", '0'))
    temperatura_ambiente = to_float(latest_sensor_data.get("temperatura-ambiente", '0.0'))
    humidity = to_float(latest_sensor_data.get("umidade", '0'))
    nivel_oxi = to_float(latest_sensor_data.get("saturacao", '0'))
    cont_tosse = to_float(latest_sensor_data.get("contagem-tosse", '0'))
    aqi = to_float(latest_sensor_data.get("qualidade-ar-aqi", '0.0'))
    pm2_5 = to_float(latest_sensor_data.get("qualidade-ar-pm25", '0.0'))
    pm10 = to_float(latest_sensor_data.get("qualidade-ar-pm10", '0.0'))

    # --- PREDIÇÃO DE CRISE ---
    resultado, acuracia, hora_predicao = 0, 0.0, hora_atual
    try:
        resultado_pred, _, acuracia_pred, _, hora_pred_api = get_user_prediction(username)
        resultado = resultado_pred or 0
        acuracia = float(acuracia_pred) if acuracia_pred else 0.0
        hora_predicao = hora_pred_api if isinstance(hora_pred_api, str) and hora_pred_api else hora_atual
    except (TypeError, ValueError, Exception) as e:
        print(f"AVISO: Erro ao obter predição do usuário '{username}': {e}.")

    # --- DADOS PARA GRÁFICOS E PAINEL ---
    labels_for_lookup = [
        (datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=4*i)).strftime("%Y-%m-%d %H:%M")
        for i in reversed(range(6))
    ]
    historical_chart_data = get_historical_data(all_sensor_data, labels_for_lookup)
    historical_chart_data['labels'] = [datetime.strptime(lbl, "%Y-%m-%d %H:%M").strftime("%H:%M") for lbl in labels_for_lookup]

    # Substituir None por valores mais recentes disponíveis ou manter None
    for metric in ["body_temp", "ambient_temp", "humidity", "spo2"]:
        historical_chart_data[metric] = [
            x if x is not None else to_float(latest_sensor_data.get({
                "body_temp": "temperatura-corporal",
                "ambient_temp": "temperatura-ambiente",
                "humidity": "umidade",
                "spo2": "saturacao"
            }[metric], 0.0)) for x in historical_chart_data.get(metric, [])
        ]

    # --- DADOS DE QUALIDADE DO AR ---
    air_quality = {
        "labels": ["AQI", "PM2.5", "PM10"],
        "values": [aqi, pm2_5, pm10]  # Usar valores numéricos diretamente
    }

    # --- RESUMO DIÁRIO PARA OS CARTÕES ---
    daily_summary = [
        {"title": "Temperatura Corporal", "value": format_string(temperatura_corporal, 1), "unit": "°C"},
        {"title": "Temperatura Ambiente", "value": format_string(temperatura_ambiente, 1), "unit": "°C"},
        {"title": "Umidade", "value": format_string(humidity, 0), "unit": "%"},
        {"title": "Saturação O₂", "value": format_string(nivel_oxi, 0), "unit": "%"},
        {"title": "Tosse (dia)", "value": format_string(cont_tosse, 0), "unit": ""}
    ]

    return render_template('painel.html',
        daily_summary=daily_summary,
        air_quality=air_quality,
        historical_chart_data=historical_chart_data,
        crisesPrediction=resultado,
        username=username,
        accuracy=acuracia,
        hora=hora_predicao,
    )




