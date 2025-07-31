import os
import logging
import concurrent.futures
from flask import Blueprint, render_template, request, jsonify
from cachetools import TTLCache
import ollama
from routes.api import get_user_location, get_air_quality, get_weather
from routes.sensores import lista_sensores
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
API_KEY_WEATHER = '7288a386509b40eb0513fd8500bd5d5d'
FAISS_INDEX_PATH = "faiss_index"

chat_bp = Blueprint('chat', __name__)
cache = TTLCache(maxsize=100, ttl=600)  # Cache de 10 minutos

# --- Funções de análise ambiental ---
def analisar_qualidade_ar(aqi):
    """Analisa o AQI e retorna estrutura de dados."""
    niveis = [
        ("Bom", "🟢", "Qualidade do ar ideal. Aproveite para atividades ao ar livre."),
        ("Razoável", "🟡", "Qualidade do ar aceitável. Pessoas sensíveis devem limitar esforços prolongados."),
        ("Moderado", "🟠", "Pessoas com asma podem sentir desconforto. Evite atividades intensas ao ar livre."),
        ("Ruim", "🔴", "Alerta de saúde. Evite exposição ao ar livre. Mantenha as janelas fechadas."),
        ("Muito Ruim", "🟣", "Risco elevado à saúde. Permaneça em ambientes internos com purificadores de ar, se possível.")
    ]
    idx = min(max(aqi - 1, 0), 4)
    nivel, emoji, recomendacao = niveis[idx]
    return {
        "condicao": "AQI",
        "valor": f"{aqi}",
        "nivel": nivel,
        "emoji": emoji,
        "recomendacao": recomendacao
    }

def analisar_umidade(humidity):
    """Analisa a umidade e retorna estrutura de dados."""
    if 30 <= humidity <= 50:
        nivel, emoji, recomendacao = "Ideal", "🟢", "Níveis de umidade ideais para a respiração."
    elif humidity > 50:
        nivel = "Alta"
        emoji = "🟡" if humidity <= 60 else "🔴"
        recomendacao = "Umidade alta pode favorecer mofo e ácaros. Considere usar um desumidificador."
    else:
        nivel = "Baixa"
        emoji = "🟡" if humidity >= 20 else "🔴"
        recomendacao = "Ar seco pode irritar as vias aéreas. Considere usar um umidificador."
    return {
        "condicao": "Umidade",
        "valor": f"{humidity}%",
        "nivel": nivel,
        "emoji": emoji,
        "recomendacao": recomendacao
    }

def analisar_temperatura(temp):
    """Analisa a temperatura e retorna estrutura de dados."""
    if 18 <= temp <= 22:
        nivel, emoji, recomendacao = "Agradável", "🟢", "Temperatura confortável, sem grandes riscos para a asma."
    elif temp < 18:
        nivel = "Fria"
        emoji = "🟡" if temp >= 12 else "🔴"
        recomendacao = "Ar frio pode causar broncoespasmo. Proteja boca e nariz ao sair."
    else:
        nivel = "Quente"
        emoji = "🟡" if temp <= 28 else "🔴"
        recomendacao = "Calor intenso pode ser um gatilho. Mantenha-se hidratado e evite o sol forte."
    return {
        "condicao": "Temperatura",
        "valor": f"{temp}°C",
        "nivel": nivel,
        "emoji": emoji,
        "recomendacao": recomendacao
    }

# --- IA Especialista (RAG) ---
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Índice '{FAISS_INDEX_PATH}' não encontrado. Execute 'criar_base_de_conhecimento.py'.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
llm = OllamaLLM(model="gemma3", temperature=0.2)

prompt_template = """
Você é um assistente de IA especialista em asma, projetado para fornecer informações claras e úteis aos pacientes.
Sua missão é responder às perguntas utilizando APENAS as informações contidas no CONTEXTO fornecido abaixo.
Não invente informações nem use conhecimento externo.
Se a resposta não estiver no contexto, diga: "Desculpe, não encontrei informações sobre isso em minha base de dados. É sempre melhor consultar um profissional de saúde para questões específicas."
Use uma linguagem simples, direta e acolhedora.
Para qualquer pergunta sobre sintomas graves (como falta de ar intensa), tratamentos ou diagnóstico, SEMPRE finalize sua resposta com a frase:
"Importante: Esta informação é para fins educativos e não substitui o aconselhamento, diagnóstico ou tratamento médico profissional. Consulte sempre o seu médico para qualquer questão de saúde."

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA (em português do Brasil):
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# --- Rotas Flask ---
@chat_bp.route("/chat")
def index():
    return render_template("chat.html")

@chat_bp.route("/api/dados-ambientais")
def dados_ambientais():
    """
    Retorna um JSON estruturado com os dados ambientais e recomendações.
    """
    try:
        lat, lon, city = get_user_location()
        cache_key = f"dados-{lat}-{lon}"
        if cache_key in cache:
            return jsonify(cache[cache_key])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_air = executor.submit(get_air_quality, lat, lon, API_KEY_WEATHER)
            future_weather = executor.submit(get_weather, lat, lon)
            aqi, pm2_5, pm10 = future_air.result()
            temperatura_amb = next((s["valor"] for s in lista_sensores if s["id"] == "temperatura-ambiente"), None)
            t, humidity = future_weather.result()

        sugestoes_list = [
            analisar_qualidade_ar(aqi),
            analisar_temperatura(temperatura_amb if temperatura_amb is not None else t),
            analisar_umidade(humidity)
        ]

        response_data = {
            "cidade": city,
            "sugestoes_ambientais": sugestoes_list
        }
        cache[cache_key] = response_data
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Erro em dados_ambientais: {str(e)}")
        return jsonify({"error": "Não foi possível obter os dados ambientais."}), 500

@chat_bp.route("/api/sugestoes-iniciais")
def sugestoes_iniciais():
    """
    Gera sugestões de perguntas comuns para pacientes asmáticos.
    """
    try:
        prompt = (
            "Sugira 3 perguntas curtas e muito comuns que um paciente com asma faria a um especialista. "
            "Responda apenas com as 3 perguntas, cada uma em uma nova linha, sem marcadores."
            "Sempre nas perguntas deixe explícito que se trata sobre um paciente asmático ou uma dúvida com relação à asma."
        )
        resposta = ollama.chat(model="gemma3", messages=[{"role": "user", "content": prompt}])['message']['content']
        sugestoes = [linha.strip() for linha in resposta.split('\n') if linha.strip()][:3]
        return jsonify({"sugestoes": sugestoes})
    except Exception as e:
        logging.error(f"Erro ao gerar sugestões: {str(e)}")
        return jsonify({"error": "Não foi possível gerar sugestões."}), 500

@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Rota principal do chatbot especialista.
    """
    data = request.get_json()
    pergunta = data.get("pergunta", "")
    if not pergunta:
        return jsonify({"error": "A pergunta não pode ser vazia."}), 400

    try:
        logging.info(f"Pergunta recebida: '{pergunta}'")
        result = qa_chain.invoke({"query": pergunta})
        resposta = result["result"]
        fontes = set(doc.metadata.get('source', 'Fonte não identificada') for doc in result.get('source_documents', []))
        if fontes:
            resposta += f"\n\n*Fontes consultadas: {', '.join(fontes)}*"
        logging.info(f"Resposta enviada: '{resposta}'")
        return jsonify({"resposta": resposta})
    except Exception as e:
        logging.error(f"Erro fatal na rota /api/chat: {str(e)}")
        return jsonify({"error": "Ocorreu um erro inesperado ao processar sua pergunta."}), 500