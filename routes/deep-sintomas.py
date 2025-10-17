import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- 1. Carregamento e Análise Exploratória dos Dados (EDA) ---
try:
    # Carrega o conjunto de dados a partir do arquivo CSV.
    df = pd.read_csv('asthma_disease_data.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'asthma_disease_data.csv' não encontrado.")
    exit()

# Remove colunas que não serão utilizadas como features para o modelo.
df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

print("--- Análise Inicial dos Dados ---")
print("Dimensões do DataFrame:", df.shape)
print("\nVerificação de valores ausentes:")
print(df.isnull().sum())

# A coluna 'Diagnosis' parece ser o nosso alvo.
# 1 indica um diagnóstico de crise, 0 indica ausência de crise.
# Vamos confirmar a distribuição para entender o desbalanceamento.
print("\nDistribuição da variável alvo (Diagnosis):")
print(df['Diagnosis'].value_counts(normalize=True))
print("-----------------------------------\n")

# --- 2. Preparação dos Dados para o Modelo ---
# Seleção de características (features) baseada em relevância clínica.
# Fontes de pesquisa indicam que sintomas respiratórios, histórico alérgico e familiar
# são preditores importantes para crises de asma. [1, 2, 8]
features = [
    'Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure',
    'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema',
    'HayFever', 'GastroesophagealReflux', 'LungFunctionFEV1', 'LungFunctionFVC',
    'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing',
    'NighttimeSymptoms', 'ExerciseInduced'
]
X = df[features]
y = df['Diagnosis'] # Alvo: 1 para crise de asma, 0 para sem crise.

# Divisão dos dados em conjuntos de treino e teste.
# 'stratify=y' é crucial para manter a mesma proporção de classes nos dois conjuntos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 3. Construção do Pipeline de Machine Learning ---
# O uso de um pipeline garante que o pré-processamento seja aplicado corretamente,
# evitando vazamento de dados (data leakage). [18, 19]

# Etapa 1: SMOTE para lidar com o desbalanceamento de classes, criando amostras sintéticas da classe minoritária. [19, 22]
# A classe RandomForestClassifier foi escolhida por sua robustez e bom desempenho em tarefas de classificação complexas. [3, 5]
# Usamos o pipeline do imblearn para integrar o SMOTE corretamente.
model_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()), # Etapa 2: Padronização das features
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1))
])

# --- 4. Treinamento e Avaliação do Modelo ---
model_pipeline.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Avaliação completa
print("\n--- Avaliação do Modelo RandomForest com SMOTE ---")
print(f"Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Sem Crise (0)', 'Crise de Asma (1)']))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Previsto Sem Crise', 'Previsto com Crise'],
            yticklabels=['Real Sem Crise', 'Real com Crise'])
plt.title('Matriz de Confusão')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.show()

# --- 5. Importância das Features ---
# Acessando o classificador treinado dentro do pipeline para extrair a importância das features.
feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Importância das Features para o Modelo ---")
print(importance_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Features Mais Importantes para Previsão de Asma')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# --- 6. Função de Previsão para Novos Pacientes ---
def prever_crise_asma(dados_paciente, modelo, colunas_modelo):
    """
    Prevê a probabilidade e a classe de uma crise de asma para um novo paciente.
    """
    # Garante que todas as colunas necessárias estão presentes
    if not all(col in dados_paciente for col in colunas_modelo):
        colunas_faltantes = [col for col in colunas_modelo if col not in dados_paciente]
        raise ValueError(f"Dados do paciente incompletos. Faltando as colunas: {colunas_faltantes}")

    paciente_df = pd.DataFrame([dados_paciente])
    
    # Reordena as colunas para corresponder à ordem do treinamento
    paciente_df = paciente_df[colunas_modelo]
    
    probabilidade_crise = modelo.predict_proba(paciente_df)[0, 1]
    classe_prevista = modelo.predict(paciente_df)[0]
    
    return probabilidade_crise, classe_prevista

# --- 7. Simulação de Previsão ---
# Exemplo 1: Paciente com baixo risco
paciente_baixo_risco = {
    'Age': 30, 'Gender': 0, 'BMI': 22, 'Smoking': 0, 'PhysicalActivity': 5,
    'DietQuality': 8, 'SleepQuality': 8, 'PollutionExposure': 2,
    'PollenExposure': 2, 'DustExposure': 2, 'PetAllergy': 0, 'FamilyHistoryAsthma': 0,
    'HistoryOfAllergies': 0, 'Eczema': 0, 'HayFever': 0, 'GastroesophagealReflux': 0,
    'LungFunctionFEV1': 4.0, 'LungFunctionFVC': 5.0, 'Wheezing': 0, 'ShortnessOfBreath': 0,
    'ChestTightness': 0, 'Coughing': 0, 'NighttimeSymptoms': 0, 'ExerciseInduced': 0
}
print("\n--- Previsão para Paciente com BAIXO Risco ---")
try:
    prob, classe = prever_crise_asma(paciente_baixo_risco, model_pipeline, features)
    print(f"Probabilidade de crise de asma: {prob*100:.2f}%")
    print(f"Classificação: {'Crise de Asma' if classe == 1 else 'Sem Crise'}")
except ValueError as e:
    print(e)

# Exemplo 2: Paciente com alto risco
paciente_alto_risco = {
    'Age': 45, 'Gender': 1, 'BMI': 28, 'Smoking': 1, 'PhysicalActivity': 2,
    'DietQuality': 4, 'SleepQuality': 5, 'PollutionExposure': 7,
    'PollenExposure': 8, 'DustExposure': 7, 'PetAllergy': 1, 'FamilyHistoryAsthma': 1,
    'HistoryOfAllergies': 1, 'Eczema': 1, 'HayFever': 1, 'GastroesophagealReflux': 1,
    'LungFunctionFEV1': 2.1, 'LungFunctionFVC': 3.5, 'Wheezing': 1, 'ShortnessOfBreath': 1,
    'ChestTightness': 1, 'Coughing': 1, 'NighttimeSymptoms': 1, 'ExerciseInduced': 1
}
print("\n--- Previsão para Paciente com ALTO Risco ---")
try:
    prob, classe = prever_crise_asma(paciente_alto_risco, model_pipeline, features)
    print(f"Probabilidade de crise de asma: {prob*100:.2f}%")
    print(f"Classificação: {'Crise de Asma' if classe == 1 else 'Sem Crise'}")
except ValueError as e:
    print(e)