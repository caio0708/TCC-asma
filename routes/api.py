import requests

# Função para obter a localização do usuário com timeout
def get_user_location():
    response = requests.get("http://ip-api.com/json/", timeout=5)
    response.raise_for_status()
    data = response.json()
    return data["lat"], data["lon"], data["city"]

    
# Função para buscar dados de qualidade do ar (Open-Meteo) com cache
def get_air_quality(lat, lon, API_KEY): # lat, lon, API_KEY
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url).json()
    aqi = response["list"][0]["main"]["aqi"]
    pm2_5 = response["list"][0]["components"]["pm2_5"]
    pm10 = response["list"][0]["components"]["pm10"]
    return aqi, pm2_5, pm10

# Função para buscar dados de clima (Open-Meteo) com cache
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    temp = data["hourly"]["temperature_2m"][0]  # Temperatura em °C
    humidity = data["hourly"]["relative_humidity_2m"][0]  # Umidade em %
    return temp, humidity
