import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Carregamento e Pré-processamento dos Dados ---

# Corrigido o nome do arquivo para o dataset fornecido.
file_path = 'dados/synthetic_asthma_dataset.csv' #corrigir nome do arquivo sintomas.csv
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    # Adicionado um sys.exit() para parar a execução se o arquivo não for encontrado.
    import sys
    sys.exit()

# Separação das features (X) e do alvo (y)
# Removido 'Patient_ID' (identificador) e 'Asthma_Control_Level' (vazamento de dados) das features.
X = df.drop(["Patient_ID", "Has_Asthma", "Asthma_Control_Level"], axis=1)
y = df["Has_Asthma"]

# Conversão de colunas categóricas em numéricas usando one-hot encoding.
# Isso é essencial para que os modelos de ML possam processar os dados.
X_encoded = pd.get_dummies(X, drop_first=True)

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Treinamento e Seleção do Melhor Modelo ---

modelos = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

trained_models = {}
model_accuracies = {}

# Treinamento e avaliação de cada modelo
print("Treinando e avaliando modelos...")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Armazenamento do modelo treinado e sua acurácia
    trained_models[nome] = modelo
    model_accuracies[nome] = accuracy
    print(f"Modelo: {nome}, Acurácia: {accuracy:.4f}")

# Seleção do melhor modelo com base na acurácia
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = trained_models[best_model_name]
best_accuracy = model_accuracies[best_model_name]

print(f"\nMelhor modelo selecionado: {best_model_name} com acurácia de {best_accuracy:.4f}")

# --- 3. Funções de Predição e Lógica da Aplicação ---

# A função de predição foi corrigida para aplicar o mesmo encoding aos novos dados
def predict_crisis(new_patient: dict, model_to_use, train_columns) -> int:
    """
    Prevê a crise de asma para um novo paciente.
    Aplica o mesmo pré-processamento (one-hot encoding) usado no treino.
    """
    df_new = pd.DataFrame([new_patient])
    df_new_encoded = pd.get_dummies(df_new)
    # Alinha as colunas com o dataset de treino, preenchendo com 0 as que faltarem
    df_new_aligned = df_new_encoded.reindex(columns=train_columns, fill_value=0)
    
    prediction = model_to_use.predict(df_new_aligned)
    return int(prediction[0])

# NOTA: As funções abaixo dependem de um ambiente Flask e de um arquivo 'sintomas.csv'.
# Elas foram corrigidas para funcionar com a nova lógica de ML, mas não podem ser executadas aqui.
def get_user_prediction(username: str) -> tuple:
    """
    Busca os dados mais recentes de um usuário e realiza a predição.
    Esta função assume a existência de um arquivo 'dados/sintomas.csv'.
    """
    # O caminho para o arquivo de sintomas precisa existir para esta função rodar.
    filepath = 'dados/sintomas.csv'

    if not os.path.isfile(filepath):
        # Retorna a acurácia do melhor modelo
        return None, f'O arquivo sintomas.csv não foi encontrado.', best_accuracy, None, None

    df_sintomas = pd.read_csv(filepath)
    if df_sintomas.empty:
        return None, 'O arquivo de sintomas está vazio.', best_accuracy, None, None

    df_user = df_sintomas[df_sintomas['Username'] == username]
    if df_user.empty:
        return None, f'Nenhum dado encontrado para o usuário {username}.', best_accuracy, None, None

    sample_patient = df_user.iloc[-1].to_dict()
    
    # Usa o melhor modelo e as colunas de treino para a predição
    resultado = predict_crisis(sample_patient, best_model, X_train.columns)

    data = sample_patient.get('Data', 'N/A')
    hora = sample_patient.get('Hora', 'N/A')

    return resultado, None, best_accuracy, data, hora


