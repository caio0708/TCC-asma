import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
import joblib

# 1) Carrega dados
data = pd.read_csv("dados/PEFR_Data_Set.csv")

# 2) Define X, y
# (removi Age e Height pois você já faz isso; lembre de incluir se forem relevantes!)
X = data.drop(columns=['Age', 'Height', 'PEFR'])
y = data['PEFR']

# 3) Divide em treino/teste (opcional, se quiser avaliar hold-out)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Define dicionário de modelos para comparação
models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),

}

# 5) Função para avaliar com cross-val R²
def evaluate_model(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name:17s} R² mean: {scores.mean():.3f}  std: {scores.std():.3f}")
    return scores.mean()

# 6) Loop de avaliação
results = {}
print("Avaliação com 5-fold CV (R²):")
for name, mdl in models.items():
    results[name] = evaluate_model(name, mdl, X_train, y_train)

# 7) Escolhe o melhor
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\n>> Melhor modelo: {best_name} (R² = {results[best_name]:.3f})")

# 8) Re-treina o melhor em todo o conjunto de treino e salva
best_model.fit(X_train, y_train)
#joblib.dump(best_model, f'PEFR_predictor_{best_name}.joblib')
print(f"Modelo salvo como PEFR_predictor_{best_name}.joblib")

#model = joblib.load('PEFR_predictor_RandomForest.joblib')

# Importa a última linha do arquivo usuarios.csv
usuarios_pessoal = pd.read_csv("dados/usuarios.csv")
ultimo_usuario = usuarios_pessoal.iloc[-1]

genero = (ultimo_usuario["Gender"])
g = 1 if genero.strip() == "Masculino" else 0

idade = int(ultimo_usuario["Age"])
altura = float(ultimo_usuario["Altura"])

usuarios_sensores = pd.read_csv("dados/sensores.csv")
usuarios_sensores = usuarios_sensores.dropna(how='all')
ultimo_sensor = usuarios_sensores.iloc[-1]

p = float(ultimo_sensor["temperatura-ambiente"])
q = float(ultimo_sensor["umidade"])
r = float(ultimo_sensor["qualidade-ar-pm25"])
s = float(ultimo_sensor["qualidade-ar-pm10"])

# Exibe os dados lidos
print("\n--- Dados do usuário (última linha de usuarios.csv) ---")
print(f"Gênero (original): {genero} -> g = {g}")
print(f"Idade: {idade} anos")
print(f"Altura: {altura} cm")
print("\n--- Dados dos sensores (última linha de sensores.csv) ---")
print(f"Temperatura: {p} °C")
print(f"Umidade: {q} %")
print(f"PM 2.5: {r}")
print(f"PM 10: {s}")

prediction = best_model.predict([[g,p,q,r,s]])
predicted_pefr = prediction[0]

if idade <18:
  pefr_ref = ((altura  - 100) * 5) + 100
else:
    
    # https://www.mdapp.co/peak-flow-calculator-76/
    if g == 1:
        pefr_ref = ( ((5.48 * (altura/100)) + 1.58 ) - (0.041 * idade) ) * 60
    elif g==0:
        pefr_ref = ((((altura/100) * 3.72) + 2,24) - (idade * 0.03)) * 60

perpefr = (predicted_pefr / pefr_ref) * 100

print(f"Predicted PEFR: {predicted_pefr:.2f}")
print(f"Reference PEFR: {pefr_ref:.2f}")
print(f"Percent of Expected: {perpefr:.2f}%")

if perpefr >= 80:
    print("SAFE")
elif perpefr >= 50:
    print("MODERATE")
else:
    print("RISK")