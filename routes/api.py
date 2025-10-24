import requests
import certifi # Importa a biblioteca certifi

# Função para obter a localização do usuário com timeout
def get_user_location():
    response = requests.get("http://ip-api.com/json/", timeout=5, verify=certifi.where()) # Adiciona verificação SSL
    response.raise_for_status()
    data = response.json()
    return data["lat"], data["lon"], data["city"]

# Função para buscar dados de qualidade do ar com tratamento de erros
def get_air_quality(lat, lon, API_KEY):
    """
    Busca dados de qualidade do ar da API OpenWeatherMap.
    Retorna uma tupla com os valores ou None em caso de falha.
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        # Adiciona um timeout de 10 segundos para a requisição
        response = requests.get(url, timeout=10, verify=certifi.where()) # Adiciona verificação SSL
        # Lança uma exceção para respostas com erro (ex: 401, 404, 500)
        response.raise_for_status()  
        data = response.json()
        # Extrai os dados
        aqi = data["list"][0]["main"]["aqi"]
        components = data["list"][0]["components"]
        pm2_5 = components.get("pm2_5", 0)
        pm10 = components.get("pm10", 0)
        o3 = components.get("o3", 0)
        no2 = components.get("no2", 0)
        so2 = components.get("so2", 0)
        
        return aqi, pm2_5, pm10, o3, no2, so2

    # Captura erros de conexão, timeout, e outros erros de requisição
    except requests.exceptions.RequestException as e:
        print(f"API.py | get_air_quality: Falha ao contatar a API de qualidade do ar: {e}")
        # Retorna None para indicar que a chamada falhou
        return None, None, None, None, None, None 

# Função para buscar dados de clima (Open-Meteo) com cache
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m"
    response = requests.get(url, timeout=5, verify=certifi.where()) # Adiciona verificação SSL
    response.raise_for_status()
    data = response.json()
    temp_api = data["hourly"]["temperature_2m"][0]  # Temperatura em °C
    humidity_api = data["hourly"]["relative_humidity_2m"][0]  # Umidade em %
    return temp_api, humidity_api
