import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# --- 1. Carregamento e Pré-processamento dos Dados ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_CRISE_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODELO_CRISE_PATH = os.path.join(MODELO_CRISE_DIR, "crise_model.joblib")

file_path = os.path.join(BASE_DIR, 'dados/synthetic_asthma_dataset.csv')

# --- Mapeamentos para Variáveis Categóricas ---
def get_categorical_mappings():
    """Retorna dicionários de mapeamento para variáveis categóricas."""
    return {
        'Gender': {'Masculino': 'Male', 'Feminino': 'Female', 'Outro': 'Other'},
        'Smoking_Status': {0: 'Never', 1: 'Former', 2: 'Current'},
        'Allergies': {0: 'None', 1: 'Dust', 2: 'Pollen', 3: 'Pets', 4: 'Multiple'},
        'Air_Pollution_Level': {0: 'Low', 1: 'Moderate', 2: 'High'},
        'Physical_Activity_Level': {0: 'Sedentary', 1: 'Moderate', 2: 'Active'},
        'Occupation_Type': {0: 'Indoor', 1: 'Outdoor'},
        'Comorbidities': {0: 'None', 1: 'Diabetes', 2: 'Hypertension', 3: 'Both'}
    }

# --- 2. Treinamento e Seleção do Melhor Modelo ---
def treinar_e_avaliar_modelos():
    """
    Carrega os dados, treina múltiplos modelos, seleciona o melhor com base na acurácia
    e salva um dicionário contendo o pipeline, as colunas e a acurácia.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo de treinamento '{file_path}' não foi encontrado.")
        return None, None, None

    X = df.drop(["Patient_ID", "Has_Asthma", "Asthma_Control_Level"], axis=1)
    y = df["Has_Asthma"]
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "Regressão Logística": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=42))]),
        "Árvore de Decisão": Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=42))]),
        "KNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
    }

    best_model, best_accuracy, best_model_name = None, 0, ""

    for nome, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Modelo: {nome}, Acurácia: {acuracia:.4f}")

        if acuracia > best_accuracy:
            best_accuracy = acuracia
            best_model = pipeline
            best_model_name = nome

    if best_model:
        print(f"\nMelhor modelo: {best_model_name} com acurácia de {best_accuracy:.4f}")
        os.makedirs(MODELO_CRISE_DIR, exist_ok=True)
        
        model_data_to_save = {
            'model': best_model,
            'columns': X_train.columns.tolist(),
            'accuracy': best_accuracy
        }
        joblib.dump(model_data_to_save, MODELO_CRISE_PATH)
        print(f"Melhor modelo e metadados salvos em: {MODELO_CRISE_PATH}")
        
    return best_model, X_train.columns, best_accuracy

# --- 3. Carregamento do Modelo e Colunas de Treino ---
try:
    print("Carregando modelo de crise existente...")
    model_data = joblib.load(MODELO_CRISE_PATH)
    best_model = model_data['model']
    X_train_cols = model_data['columns']
    best_accuracy = model_data['accuracy']
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Modelo de crise não encontrado. Treinando um novo modelo...")
    best_model, X_train_cols, best_accuracy = treinar_e_avaliar_modelos()
except Exception as e:
    print(f"Um erro inesperado ocorreu ao carregar o modelo: {e}")
    print("Tentando treinar um novo modelo...")
    best_model, X_train_cols, best_accuracy = treinar_e_avaliar_modelos()

# --- 4. Função de Predição ---
def predict_crisis(sample_patient: dict, model, expected_cols: list) -> int:
    """
    Realiza a predição para um novo paciente, alinhando as colunas
    com as colunas usadas no treinamento.
    """
    df_new = pd.DataFrame([sample_patient])
    
    # Remove colunas que não são features
    df_new = df_new.drop(columns=['Data', 'Hora', 'Username'], errors='ignore')

    df_new_encoded = pd.get_dummies(df_new)
    df_new_aligned = df_new_encoded.reindex(columns=expected_cols, fill_value=0)
    
    prediction = model.predict(df_new_aligned)
    return int(prediction[0])

def get_user_prediction(username: str) -> tuple:
    """
    Busca os dados mais recentes de um usuário e realiza a predição.
    """
    filepath = os.path.join(BASE_DIR, 'dados/sintomas.csv')

    if not os.path.isfile(filepath):
        return None, f'O arquivo sintomas.csv não foi encontrado.', best_accuracy, None, None

    try:
        df_sintomas = pd.read_csv(filepath)
        if df_sintomas.empty:
            return None, 'O arquivo de sintomas está vazio.', best_accuracy, None, None
    except pd.errors.EmptyDataError:
        return None, 'O arquivo de sintomas está vazio.', best_accuracy, None, None

    df_user = df_sintomas[df_sintomas['Username'] == username]
    if df_user.empty:
        return None, f'Nenhum dado encontrado para o usuário {username}.', best_accuracy, None, None

    sample_patient = df_user.iloc[-1].to_dict()

    try:
        resultado = predict_crisis(sample_patient, best_model, X_train_cols)
    except ValueError as e:
        return None, f"Erro de valor ao alinhar os dados: {e}", best_accuracy, None, None
    
    return resultado, None, best_accuracy, sample_patient.get("Data"), sample_patient.get("Hora")