import os
import sys
import pandas as pd
from flask import Blueprint, session, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timezone

# ---------- Treina o modelo e calcula a acurácia ----------
df = pd.read_csv('dados/treinamento_sintomas.csv')
label_cols = ['Severity_Mild', 'Severity_Moderate', 'Severity_None']
df['Crisis'] = df['Severity_Mild'] + df['Severity_Moderate']
df = df.drop(columns=label_cols)
X = df.drop(columns=['Crisis'])
y = df['Crisis']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelos = {
    "Random Forest": RandomForestClassifier(),
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Árvore de Decisão": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# Treinamento e avaliação de cada modelo
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

def predict_crisis(new_patient: dict) -> int:
    df_new = pd.DataFrame([new_patient])
    df_new = df_new.reindex(columns=X_train.columns, fill_value=0)
    return int(modelo.predict(df_new)[0])

def get_user_prediction(username: str) -> tuple:
    filepath = 'dados/sintomas.csv'

    if not os.path.isfile(filepath):
        return None, f'O arquivo sintomas.csv não foi encontrado para o usuário {username}', accuracy, None, None

    df_sintomas = pd.read_csv(filepath)
    if df_sintomas.empty:
        return None, 'O arquivo de sintomas está vazio', accuracy, None, None

    # Filtrar os dados pelo usuário
    df_user = df_sintomas[df_sintomas['Username'] == username]
    if df_user.empty:
        return None, f'Nenhum dado encontrado para o usuário {username}', accuracy, None, None

    # Pegar a última linha para o usuário
    sample_patient = df_user.iloc[-1].to_dict()
    
    resultado = predict_crisis(sample_patient)

    data = sample_patient['Data']
    hora = sample_patient['Hora']

    return resultado, None, accuracy, data, hora

def prever_crise():
    username = session.get('username')
    if not username:
        return jsonify({'erro': 'Usuário não autenticado'}), 401

    resultado, erro, acuracia, data, hora = get_user_prediction(username)
    if erro:
        status_code = 404 if 'não foi encontrado' in erro else 400
        return jsonify({'erro': erro}), status_code

    return jsonify({
        'usuario': username,
        'crise': resultado,
        'acuracia': acuracia,
        'data': data,
        'hora': hora,
    })