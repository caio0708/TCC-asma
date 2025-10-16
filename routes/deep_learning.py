import os
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Cria diretório para salvar modelos
os.makedirs('modelos', exist_ok=True)

# Função genérica de treino e salvamento de modelo clássico

def treinar_salvar_classico(X_train, y_train, X_val, y_val, pipeline, param_dist,
                             model_name, cv_splits=5, scoring='roc_auc', n_iter=20):
    min_samples = y_train.value_counts().min()
    n_splits = min(cv_splits, min_samples) if min_samples >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist,
        n_iter=n_iter, scoring=scoring, cv=cv,
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    # Calibra probabilidades
    calibrator = CalibratedClassifierCV(best_pipe, cv='prefit')
    calibrator.fit(X_val, y_val)

    # Avaliação
    probas = calibrator.predict_proba(X_val)[:, 1]
    preds = (probas >= 0.5).astype(int)
    print(f"--- Avaliação {model_name} ---")
    print(f"ROC-AUC val: {roc_auc_score(y_val, probas):.4f}")
    print(f"F1-score val: {f1_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds))

    # Salva modelo calibrado
    joblib.dump(calibrator, os.path.join('modelos', f'{model_name}.pkl'))
    return calibrator

    # Treina modelo de sintomas

#  df_sint = pd.read_csv('dados/treinamento_sintomas.csv')
#  df_sint['asma'] = np.where(df_sint['Severity_None'] == 1, 0, 1)
#  cols_sint = ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
#              'Pains', 'Sore-Throat', 'Nasal-Congestion', 'Runny-Nose']
#  X_s, y_s = df_sint[cols_sint], df_sint['asma']
#  X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(X_s, y_s,
#                                                      test_size=0.2,
#                                                      stratify=y_s,
#                                                      random_state=42)
#
#  pipe_sint = ImbPipeline([
#      ('smote', SMOTE(random_state=42)),
#      ('scaler', StandardScaler()),
#      ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
#  ])
#  param_sint = {
#      'clf__n_estimators': [100, 200, 300],
#      'clf__max_depth': [None, 10, 20, 30],
#      'clf__min_samples_split': [2, 5, 10],
#      'clf__min_samples_leaf': [1, 2, 4]
#  }
#  model_sintomas = treinar_salvar_classico(
#      X_tr_s, y_tr_s, X_val_s, y_val_s,
#      pipe_sint, param_sint,
#      'rf_sintomas'
#  ) 

# Treina modelo de sensores (rede neural)

df_sens = pd.read_csv('dados/treinamento_sensores.csv')
sensor_cols = [
    'frequencia-respiratoria_mean', 'batimentos-cardiacos_mean', 'saturacao_mean',
    'temperatura_mean', 'qualidade-ar-pm25_mean', 'qualidade-ar-pm10_mean',
    'qualidade-ar-aqi_mean', 'movimento-toracico_mean', 'contagem-tosse_mean', 'umidade_mean'
]
# Garante colunas
for col in sensor_cols:
    if col not in df_sens.columns:
        df_sens[col] = 0
if 'target' not in df_sens.columns:
    df_sens['target'] = 0  # placeholder

X_se = df_sens[sensor_cols].values
y_se = df_sens['target'].values
# Balanceamento
sm = SMOTE(random_state=42)
try:
    X_res, y_res = sm.fit_resample(X_se, y_se)
except ValueError:
    X_res, y_res = X_se, y_se

X_tr_se, X_val_se, y_tr_se, y_val_se = train_test_split(
    X_res, y_res, test_size=0.2,
    stratify=y_res, random_state=42
)

scaler_sens = StandardScaler()
X_tr_se = scaler_sens.fit_transform(X_tr_se)
X_val_se = scaler_sens.transform(X_val_se)

model_sens = Sequential([
    Input(shape=(len(sensor_cols),)),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.001, 0.001)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_sens.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

ckpt = ModelCheckpoint('modelos/best_sensores.keras', save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model_sens.fit(
    X_tr_se, y_tr_se,
    validation_data=(X_val_se, y_val_se),
    epochs=150, batch_size=32,
    callbacks=[es, ckpt]
)

# Salva modelo de sensores e scaler
model_sens.save('modelos/model_sensores.keras')
joblib.dump(scaler_sens, 'modelos/scaler_sensores.pkl')
joblib.dump(sensor_cols, 'modelos/sensor_cols.pkl')

# Funções de pré-processamento e predição

def preprocess_paciente(sensor_path, sintomas_path):
    # Sensores
    df_s = pd.read_csv(sensor_path)
    sens_means = df_s[[
        'frequencia-respiratoria', 'batimentos-cardiacos', 'saturacao',
        'temperatura', 'qualidade-ar-pm25', 'qualidade-ar-pm10',
        'qualidade-ar-aqi', 'movimento-toracico', 'contagem-tosse', 'umidade'
    ]].mean()
    vec_s = sens_means.values.reshape(1, -1)
    vec_s_scaled = scaler_sens.transform(vec_s)

    # Sintomas
 #   df_q = pd.read_csv(sintomas_path)
  #  sym_vals = [int(df_q.get(col, pd.Series([0])).iloc[0] >= 1) for col in cols_sint]
  #  vec_sym = np.array(sym_vals).reshape(1, -1)

    return vec_s_scaled


def prever_risco(sensor_path, sintomas_path, age=None, gender=None):
    vec_s = preprocess_paciente(sensor_path, sintomas_path)
    #prob_sym = model_sintomas.predict_proba(vec_sym)[0, 1]
    prob_sens = float(model_sens.predict(vec_s)[0, 0])
    prob_tot =  0.4 * prob_sens
    if prob_tot >= 0.7:
        risco = 'Alto Risco'
    elif prob_tot >= 0.4:
        risco = 'Médio Risco'
    else:
        risco = 'Baixo Risco'
    return round(prob_sens, 2), round(prob_tot, 2), risco



# Teste com dados de exemplo
try:
    print("\n--- Teste com dados de exemplo ---")
    prob_sens, prob_total, risco = prever_risco(
        'dados/sensores.csv', 'dados/sintomas.csv', age=30, gender='Female'
    )
    #rint(f"Probabilidade Sintomas: {prob_sym:.2f}")
    print(f"Probabilidade Sensores: {prob_sens:.2f}")
    print(f"Probabilidade Total: {prob_total:.2f}")
    print(f"Risco: {risco}")
except Exception as e:
    print(f"Erro durante o teste: {e}")

print("\nTreinamento e avaliação concluídos com sucesso!")