import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import joblib
import sqlite3 # Importado para ler o banco de dados

warnings.filterwarnings('ignore')

# --- Passo 1: Carregamento e Preparação dos Dados ---
print("Carregando dados...")
try:
    # --- ALTERAÇÃO: LER DADOS DO BANCO DE DADOS SQLITE ---
    print("Conectando ao banco de dados 'dados/sensores.db'...")
    conn = sqlite3.connect('dados/sensores.db')
    # Carrega a tabela 'sensores' inteira para um DataFrame
    df_sensores = pd.read_sql_query("SELECT * FROM sensores", conn)
    conn.close()
    print(f"Dados dos sensores carregados do banco. {len(df_sensores)} registros encontrados.")
    # --- FIM DA ALTERAÇÃO ---

    # Carrega os outros arquivos CSV como antes
    df_usuarios = pd.read_csv("dados/usuarios.csv")
    df_crises = pd.read_csv("dados/registro_crise.csv")
    print("Arquivos de usuários e crises carregados com sucesso.")

except (sqlite3.Error, FileNotFoundError) as e:
    print(f"Erro ao carregar os dados. Verifique se os arquivos 'sensores.db', 'usuarios.csv' e 'registro_crise.csv' existem. Detalhes: {e}")
    exit()


# --- Passo 2: Pré-processamento e Junção dos Dados ---
# (Esta seção permanece a mesma, pois a lógica de processamento dos dados não muda)
print("\nIniciando pré-processamento e junção dos dados...")
df_sensores['timestamp'] = pd.to_datetime(df_sensores['Data'] + ' ' + df_sensores['Hora'])
df_sensores = df_sensores.sort_values('timestamp')

df_crises['timestamp'] = pd.to_datetime(df_crises['Timestamp'])
df_crises = df_crises.sort_values('timestamp')

usuario_com_crise = df_crises['Username'].unique()[0]
dados_usuario = df_usuarios[df_usuarios['Username'] == usuario_com_crise].iloc[0]

print("Expandindo a janela de eventos de crise...")
df_sensores['crisis_event'] = 0
for crise_time in df_crises['timestamp']:
    start_window = crise_time - pd.Timedelta('15min')
    end_window = crise_time
    indices_in_window = df_sensores[(df_sensores['timestamp'] >= start_window) & (df_sensores['timestamp'] <= end_window)].index
    df_sensores.loc[indices_in_window, 'crisis_event'] = 1

num_crises = df_sensores['crisis_event'].sum()
print(f"Associados {num_crises} pontos de dados a eventos de crise (usando janela de 15 min).")

df_sensores.rename(columns={
    'frequencia-respiratoria': 'respiratory_rate', 'batimentos-cardiacos': 'heart_rate',
    'saturacao': 'spo2', 'temperatura-corporal': 'body_temp', 'qualidade-ar-aqi': 'aqi',
    'contagem-tosse': 'cough_count', 'umidade': 'humidity'
}, inplace=True)
df_sensores['activity_level'] = np.sqrt(df_sensores['acelerometro-x']**2 + df_sensores['acelerometro-y']**2 + df_sensores['acelerometro-z']**2)
df_sensores['user_id'] = dados_usuario['Username']
df_sensores['age'] = dados_usuario['Age']
df_sensores['gender'] = 1 if str(dados_usuario['Gender']).strip().lower() == 'masculino' else 0
df_sensores['height'] = dados_usuario['Altura']

colunas_finais = [
    'timestamp', 'user_id', 'respiratory_rate', 'heart_rate', 'spo2', 'body_temp', 'aqi',
    'cough_count', 'humidity', 'activity_level', 'age', 'gender', 'height', 'crisis_event'
]
df = df_sensores[colunas_finais].copy().dropna()
print("Dados preparados e unidos com sucesso. Shape final:", df.shape)

# --- Passo 3: Engenharia de Features (Feature Engineering) ---
# (Esta função permanece a mesma)
def create_features(data):
    df_featured = data.copy()
    df_featured = df_featured.sort_values(by=['user_id', 'timestamp']).set_index('timestamp')
    sensor_metrics = ['spo2', 'heart_rate', 'cough_count', 'body_temp', 'respiratory_rate', 'humidity', 'aqi', 'activity_level']
    windows = ['10min', '30min', '60min']
    for metric in sensor_metrics:
        for window in windows:
            df_featured[f'{metric}_avg_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_featured[f'{metric}_std_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).std())
            df_featured[f'{metric}_max_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).max())
    df_featured.fillna(0, inplace=True)
    return df_featured.reset_index()

print("\nCriando features a partir dos dados combinados...")
df_featured = create_features(df)
print("Features criadas com sucesso. Shape:", df_featured.shape)

# --- Passo 4 e 5: Treinamento e Avaliação do Modelo ---
# (Esta seção permanece a mesma)
if not df_featured.empty and df_featured['crisis_event'].sum() > 1:
    print("\nIniciando o treinamento do modelo com Validação Cruzada...")

    features = [col for col in df_featured.columns if col not in ['timestamp', 'user_id', 'crisis_event']]
    X = df_featured[features]
    y = df_featured['crisis_event']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_preds = np.zeros(len(df_featured))
    scale_pos_weight_value = y.value_counts()[0] / y.value_counts()[1] if 1 in y.value_counts() and len(y.value_counts()) > 1 else 1
    print(f"Scale Pos Weight (para balanceamento de classes): {scale_pos_weight_value:.2f}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Treinando Fold {fold+1}/{skf.get_n_splits()} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        lgbm = lgb.LGBMClassifier(
            objective='binary', metric='auc', random_state=42, n_jobs=-1,
            scale_pos_weight=scale_pos_weight_value, n_estimators=1000
        )
        
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        val_preds = lgbm.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        models.append(lgbm)

    model_final = models[-1] 
    joblib.dump(model_final, 'modelo_crise_asma.pkl')
    print("\nModelo final salvo como 'modelo_crise_asma.pkl'")

    print("\nAvaliação geral do modelo (usando predições out-of-fold):")
    y_pred = (oof_preds > 0.5).astype(int)
    print("\nRelatório de Classificação:")
    print(classification_report(y, y_pred, zero_division=0))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y, y_pred))

else:
    print("\nTreinamento não pode ser realizado. DataFrame vazio ou sem eventos de crise suficientes.")
    model_final = None

# --- Função para Calcular Score de Risco ---
# (Esta função permanece a mesma)
def calculate_risk_score(live_data_df, model):
    if len(live_data_df) < 1:
        raise ValueError("É necessário pelo menos 1 registro de dados para calcular as features.")
    
    if model is None:
        try:
            model = joblib.load('modelo_crise_asma.pkl')
        except FileNotFoundError:
            print("ERRO: Modelo 'modelo_crise_asma.pkl' não encontrado. Execute o treinamento primeiro.")
            return 0, 0
            
    live_features_df = create_features(live_data_df)
    if live_features_df.empty: return 0, 0
    
    latest_features = live_features_df.tail(1)
    model_features = model.booster_.feature_name()
    latest_features = latest_features.reindex(columns=model_features, fill_value=0)

    crisis_probability = model.predict_proba(latest_features[model_features])[:, 1][0]
    risk_score = crisis_probability * 10
    return risk_score, crisis_probability

# --- ALTERAÇÃO: ANÁLISE DA ÚLTIMA LINHA REAL DO BANCO DE DADOS ---
if model_final:
    print("\n--- Análise da Última Leitura Real do Banco de Dados ---")
    
    # Pega os últimos 120 registros para ter dados suficientes para as features de janela
    live_data_for_features = df.tail(120).copy() 
    
    if not live_data_for_features.empty:
        try:
            # Calcula o score usando os dados reais, sem simulação
            score, prob = calculate_risk_score(live_data_for_features, model_final)
            
            # Pega a última linha para exibir os dados exatos que foram analisados
            last_real_row = live_data_for_features.tail(1)
            
            print(f"Dados reais do paciente (última leitura do banco de dados):")
            print(last_real_row[['timestamp', 'spo2', 'heart_rate', 'respiratory_rate']])
            print(f"\nProbabilidade de Crise Calculada: {prob:.2%}")
            print(f"SCORE DE RISCO (0 a 10): {score:.2f}")

            if score > 7: print("ALERTA: Risco de crise elevado! Monitorar o paciente de perto.")
            elif score > 4: print("AVISO: Risco moderado. Verificar dados ambientais e sintomas.")
            else: print("NORMAL: Risco baixo.")
            
        except Exception as e:
            print(f"Não foi possível calcular o score. Erro: {e}")
    else:
        print("Não foi possível realizar a análise final por falta de dados contínuos.")
# --- FIM DA ALTERAÇÃO ---

# import pandas as pd

# import numpy as np

# import lightgbm as lgb

# from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import warnings

# import joblib # Para salvar o modelo



# warnings.filterwarnings('ignore')



# # --- Passo 1: Carregamento e Preparação dos Dados ---

# print("Carregando todos os arquivos de dados...")

# try:

#     df_sensores = pd.read_csv("dados/sensores.csv")

#     df_usuarios = pd.read_csv("dados/usuarios.csv")

#     df_crises = pd.read_csv("dados/registro_crise.csv")

#     print("Arquivos carregados com sucesso.")

# except FileNotFoundError as e:

#     print(f"Erro: Arquivo não encontrado. Verifique se 'sensores.csv', 'usuarios.csv' e 'registro_crise.csv' estão no lugar certo. Detalhes: {e}")

#     exit()



# # --- Passo 2: Pré-processamento e Junção dos Dados ---

# print("\nIniciando pré-processamento e junção dos dados...")

# df_sensores['timestamp'] = pd.to_datetime(df_sensores['Data'] + ' ' + df_sensores['Hora'])

# df_sensores = df_sensores.sort_values('timestamp')



# df_crises['timestamp'] = pd.to_datetime(df_crises['Timestamp'])

# df_crises = df_crises.sort_values('timestamp')



# usuario_com_crise = df_crises['Username'].unique()[0]

# print(f"Usuário com crises identificadas: '{usuario_com_crise}'")

# dados_usuario = df_usuarios[df_usuarios['Username'] == usuario_com_crise].iloc[0]



# # --- ALTERAÇÃO 1: EXPANDIR A JANELA DA CRISE ---

# # Em vez de marcar apenas um ponto, marcaremos um período antes de cada crise.

# print("Expandindo a janela de eventos de crise...")

# df_sensores['crisis_event'] = 0

# for crise_time in df_crises['timestamp']:

#     # Define o início e o fim da janela (ex: 15 minutos antes da crise)

#     start_window = crise_time - pd.Timedelta('15min')

#     end_window = crise_time

#    

#     # Encontra os índices dos dados de sensores dentro dessa janela

#     indices_in_window = df_sensores[(df_sensores['timestamp'] >= start_window) & (df_sensores['timestamp'] <= end_window)].index

#    

#     # Marca esses índices como evento de crise (1)

#     df_sensores.loc[indices_in_window, 'crisis_event'] = 1



# num_crises = df_sensores['crisis_event'].sum()

# print(f"Associados {num_crises} pontos de dados a eventos de crise (usando janela de 15 min).")

# # --- FIM DA ALTERAÇÃO 1 ---



# # Continuação do pré-processamento...

# df_sensores.rename(columns={

#     'frequencia-respiratoria': 'respiratory_rate', 'batimentos-cardiacos': 'heart_rate',

#     'saturacao': 'spo2', 'temperatura-corporal': 'body_temp', 'qualidade-ar-aqi': 'aqi',

#     'contagem-tosse': 'cough_count', 'umidade': 'humidity'

# }, inplace=True)

# df_sensores['activity_level'] = np.sqrt(df_sensores['acelerometro-x']**2 + df_sensores['acelerometro-y']**2 + df_sensores['acelerometro-z']**2)

# df_sensores['user_id'] = dados_usuario['Username']

# df_sensores['age'] = dados_usuario['Age']

# df_sensores['gender'] = 1 if str(dados_usuario['Gender']).strip().lower() == 'masculino' else 0

# df_sensores['height'] = dados_usuario['Altura']



# colunas_finais = [

#     'timestamp', 'user_id', 'respiratory_rate', 'heart_rate', 'spo2', 'body_temp', 'aqi',

#     'cough_count', 'humidity', 'activity_level', 'age', 'gender', 'height', 'crisis_event'

# ]

# df = df_sensores[colunas_finais].copy().dropna()

# print("Dados preparados e unidos com sucesso. Shape final:", df.shape)



# # --- Passo 3: Engenharia de Features (Feature Engineering) ---

# def create_features(data):

#     df_featured = data.copy()

#     df_featured = df_featured.sort_values(by=['user_id', 'timestamp']).set_index('timestamp')

#     sensor_metrics = ['spo2', 'heart_rate', 'cough_count', 'body_temp', 'respiratory_rate', 'humidity', 'aqi', 'activity_level']

#     windows = ['10min', '30min', '60min']

#     for metric in sensor_metrics:

#         for window in windows:

#             df_featured[f'{metric}_avg_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).mean())

#             df_featured[f'{metric}_std_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).std())

#             df_featured[f'{metric}_max_{window}'] = df_featured.groupby('user_id')[metric].transform(lambda x: x.rolling(window, min_periods=1).max())

#     df_featured.fillna(0, inplace=True)

#     return df_featured.reset_index()



# print("\nCriando features a partir dos dados combinados...")

# df_featured = create_features(df)

# print("Features criadas com sucesso. Shape:", df_featured.shape)



# # --- Passo 4 e 5: Treinamento e Avaliação do Modelo ---

# if not df_featured.empty and df_featured['crisis_event'].sum() > 1: # Precisa de pelo menos 2 exemplos para estratificar

#     print("\nIniciando o treinamento do modelo com Validação Cruzada...")



#     features = [col for col in df_featured.columns if col not in ['timestamp', 'user_id', 'crisis_event']]

#     X = df_featured[features]

#     y = df_featured['crisis_event']



#     # --- ALTERAÇÃO 2: USAR VALIDAÇÃO CRUZADA ESTRATIFICADA ---

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#    

#     models = []

#     oof_preds = np.zeros(len(df_featured))

#    

#     scale_pos_weight_value = y.value_counts()[0] / y.value_counts()[1] if 1 in y.value_counts() and len(y.value_counts()) > 1 else 1

#     print(f"Scale Pos Weight (para balanceamento de classes): {scale_pos_weight_value:.2f}")



#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

#         print(f"\n--- Treinando Fold {fold+1}/{skf.get_n_splits()} ---")

#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

#         X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]



#         lgbm = lgb.LGBMClassifier(

#             objective='binary', metric='auc', random_state=42, n_jobs=-1,

#             scale_pos_weight=scale_pos_weight_value, n_estimators=1000

#         )

#        

#         lgbm.fit(

#             X_train, y_train,

#             eval_set=[(X_val, y_val)],

#             eval_metric='auc',

#             callbacks=[lgb.early_stopping(100, verbose=False)] # verbose=False para um log mais limpo

#         )

#        

#         val_preds = lgbm.predict_proba(X_val)[:, 1]

#         oof_preds[val_idx] = val_preds

#         models.append(lgbm)



#     # Salva o melhor modelo (ou o último, para simplificar)

#     model_final = models[-1]

#     joblib.dump(model_final, 'modelo_crise_asma.pkl')

#     print("\nModelo final salvo como 'modelo_crise_asma.pkl'")



#     print("\nAvaliação geral do modelo (usando predições out-of-fold):")

#     y_pred = (oof_preds > 0.5).astype(int)

#     print("\nRelatório de Classificação:")

#     print(classification_report(y, y_pred, zero_division=0))

#     print("\nMatriz de Confusão:")

#     print(confusion_matrix(y, y_pred))

#     # --- FIM DA ALTERAÇÃO 2 ---



# else:

#     print("\nTreinamento não pode ser realizado. DataFrame vazio ou sem eventos de crise suficientes.")

#     model_final = None # Garante que a variável exista



# # --- Função para Calcular Score de Risco ---

# def calculate_risk_score(live_data_df, model):

#     if len(live_data_df) < 1:

#         raise ValueError("É necessário pelo menos 1 registro de dados para calcular as features.")

#    

#     # Carrega o modelo se não foi passado

#     if model is None:

#         try:

#             model = joblib.load('modelo_crise_asma.pkl')

#         except FileNotFoundError:

#             print("ERRO: Modelo 'modelo_crise_asma.pkl' não encontrado. Execute o treinamento primeiro.")

#             return 0, 0

#            

#     live_features_df = create_features(live_data_df)

#     if live_features_df.empty: return 0, 0

#    

#     latest_features = live_features_df.tail(1)

#     model_features = model.booster_.feature_name()

#     latest_features = latest_features.reindex(columns=model_features, fill_value=0) # Garante todas as colunas



#     crisis_probability = model.predict_proba(latest_features[model_features])[:, 1][0]

#     risk_score = crisis_probability * 10

#     return risk_score, crisis_probability



# # --- Simulação de Uso em Tempo Real ---

# if model_final: # Apenas executa se o modelo foi treinado

#     print("\n--- Simulação de Uso em Tempo Real ---")

#     sample_live_data = df.tail(120).copy()

#     if not sample_live_data.empty:

#         last_row_index = sample_live_data.index[-1]

#         sample_live_data.loc[last_row_index, 'spo2'] = 93.5

#         sample_live_data.loc[last_row_index, 'heart_rate'] = 115

#         sample_live_data.loc[last_row_index, 'respiratory_rate'] = 25

#         try:

#             score, prob = calculate_risk_score(sample_live_data, model_final)

#             print(f"Dados do paciente em tempo real (última leitura simulada):")

#             print(sample_live_data.tail(1)[['spo2', 'heart_rate', 'respiratory_rate']])

#             print(f"\nProbabilidade de Crise Calculada: {prob:.2%}")

#             print(f"SCORE DE RISCO (0 a 10): {score:.2f}")



#             if score > 7: print("ALERTA: Risco de crise elevado! Monitorar o paciente de perto.")

#             elif score > 4: print("AVISO: Risco moderado. Verificar dados ambientais e sintomas.")

#             else: print("NORMAL: Risco baixo.")

#         except Exception as e:

#             print(f"Não foi possível calcular o score em tempo real. Erro: {e}")

#     else:

#         print("Não foi possível gerar um exemplo em tempo real por falta de dados contínuos.")