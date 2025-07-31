from flask import Blueprint, render_template, request, redirect, jsonify, session, url_for, flash
from datetime import datetime, timedelta
from routes.sensores import lista_sensores
import os
import csv
import joblib
import tensorflow as tf
from routes.api import get_weather, get_air_quality
import pandas as pd

from routes.crises import get_user_prediction, accuracy

painel_bp = Blueprint('painel', __name__)

# Coordenadas fixas
LAT, LON = -23.5505, -46.6333
API_KEY = '7288a386509b40eb0513fd8500bd5d5d' 

def read_symptoms_from_csv():
    sintomas = []
    caminho = 'dados/sintomas.csv'
    if os.path.isfile(caminho):
        with open(caminho, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sintomas = list(reader)
    return sintomas

def align_symptoms_with_timestamps_detalhado(sintomas, target_labels):
    resultado = []
    for label in target_labels:
        encontrados = []
        for s in sintomas:
            try:
                dt = datetime.strptime(f"{s['Data']} {s['Hora']}", "%Y-%m-%d %H:%M:%S")
                if dt.strftime("%H:%M") == label:
                    intens = {
                        "Tosse": int(s.get("Tosse") or 0),
                        "Chiado": int(s.get("Chiado") or 0),
                        "Falta de ar": int(s.get("Falta de ar") or 0),
                        "Aperto no peito": int(s.get("Aperto no peito") or 0)
                    }
                    mx_sint, mx_val = max(intens.items(), key=lambda x: x[1])
                    if mx_val > 0:
                        encontrados.append((mx_sint, mx_val))
            except Exception:
                continue

        if encontrados:
            sint, val = max(encontrados, key=lambda x: x[1])
            resultado.append({"hora": label, "sintoma": sint, "intensidade": val})
        else:
            resultado.append({"hora": label, "sintoma": "", "intensidade": 0})

    return resultado

@painel_bp.route('/')
def painel():
    # 1) Sensores
    freq_resp  = next(s["valor"] for s in lista_sensores if s["id"] == "frequencia-respiratoria")
    nivel_oxi  = next(s["valor"] for s in lista_sensores if s["id"] == "saturacao")
    cont_tosse = next(s["valor"] for s in lista_sensores if s["id"] == "contagem-tosse")

    # 2) APIs
    temp, humidity = get_weather(LAT, LON)
    aqi, pm2_5, pm10 = get_air_quality(LAT, LON, API_KEY)

    # Garantir que são números válidos
    try:
        aqi = float(aqi)
        pm2_5 = float(pm2_5)
        pm10 = float(pm10)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Erro: valores de get_air_quality inválidos - {e}")

    # 3) Sintomas históricos (últimas 6 medições)
    sintomas = read_symptoms_from_csv()
    labels = [
        (datetime.now().replace(minute=0, second=0) - timedelta(hours=4*i)).strftime("%H:%M")
        for i in reversed(range(6))
    ]
    intensidades = align_symptoms_with_timestamps_detalhado(sintomas, labels)

    caminho = 'dados/sintomas.csv'
    crisis_probs = get_user_prediction(caminho)
    #crisis_probs = 0

    # 4) Predição de crise
    username = session.get('username')

    resultado, erro, acuracia, data, hora = get_user_prediction(username)

    # 5) Contexto para template
    air_quality = {
        "labels": ["AQI", "PM2.5", "PM10"],
        "values": [aqi, pm2_5, pm10]
    }
    umid_vs_symptoms = {
        "labels": labels,
        "umid_quality": [humidity] * len(labels),
        "symptoms": intensidades
    }

    return render_template('painel.html',
        daily_summary = [
            {"title": "Temperatura", "value": f"{temp:.1f}", "unit": "°C"},
            {"title": "Umidade",    "value": f"{humidity}%", "unit": ""},
            {"title": "Freq. Resp.", "value": freq_resp, "unit": "rpm"},
            {"title": "Tosse (dia)", "value": cont_tosse, "unit": ""}
        ],
        air_quality = air_quality,
        umid_vs_symptoms = umid_vs_symptoms,
        crisesPrediction = resultado ,
        username = username ,
        accuracy = acuracia ,
        hora = hora ,
    )

@painel_bp.route('/api/graficos')
def api_graficos():
    temp, humidity = get_weather(LAT, LON)
    aqi, pm2_5, pm10 = get_air_quality(LAT, LON, API_KEY)

    try:
        aqi = float(aqi)
        pm2_5 = float(pm2_5)
        pm10 = float(pm10)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Erro: valores de get_air_quality inválidos - {e}")

    sintomas = read_symptoms_from_csv()
    labels = [
        (datetime.now().replace(minute=0, second=0) - timedelta(hours=4*i)).strftime("%H:%M")
        for i in reversed(range(6))
    ]
    symptoms = align_symptoms_with_timestamps_detalhado(sintomas, labels)

    data = {
        "weather": {"temp": temp, "humidity": humidity},
        "air_quality": {"aqi": aqi, "pm2_5": pm2_5, "pm10": pm10},
        "symptoms_history": symptoms,
    }

    return jsonify(data)

@painel_bp.route("/questionario")
def questionario():
    return render_template("questionario.html")

@painel_bp.route("/salvar_sintomas", methods=["POST"])
def salvar_sintomas():
    if request.method == 'POST':
        if 'username' not in session:
            flash('Faça login primeiro.', 'warning')
            return redirect(url_for('configuracoes.configuracoes'))

        data = request.form.get('data') or datetime.now().strftime('%Y-%m-%d')
        hora = request.form.get('hora') or datetime.now().strftime('%H:%M')

        def sintoma(nome):
            vals = [int(v) for v in request.form.getlist(nome)]
            return max(vals) if vals else 0

        # coletar todos os sintomas
        sintomas_data = [
            data, hora,
            sintoma("Tiredness"), sintoma("Dry-Cough"), sintoma("Difficulty-in-Breathing"),
            sintoma("Sore-Throat"), sintoma("None_Sympton"), sintoma("Pains"),
            sintoma("Nasal-Congestion"), sintoma("Runny-Nose"), sintoma("None_Experiencing"),
            # encoding idade e gênero
            1 if 1 <= session.get('age',0) <= 9 else 0,
            1 if 10 <= session.get('age',0) <= 19 else 0,
            1 if 20 <= session.get('age',0) <= 24 else 0,
            1 if 25 <= session.get('age',0) <= 59 else 0,
            1 if session.get('age',0) >= 60 else 0,
            1 if session.get('gender','') == "Feminino" else 0,
            1 if session.get('gender','') == "Masculino" else 0,
            session.get('username','')
        ]

        file_exists = os.path.isfile("dados/sintomas.csv")
        with open("dados/sintomas.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Data","Hora","Tiredness","Dry-Cough","Difficulty-in-Breathing",
                    "Sore-Throat","None_Sympton","Pains","Nasal-Congestion",
                    "Runny-Nose","None_Experiencing","Age_0-9","Age_10-19",
                    "Age_20-24","Age_25-59","Age_60+","Gender_Female","Gender_Male","Username"
                ])
            writer.writerow(sintomas_data)

        return redirect("/")