import os
import logging
import asyncio
from functools import lru_cache
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
FAISS_INDEX_PATH = "faiss_index" # Pasta com as fontes para IA aprender

chat_bp = Blueprint('chat', __name__)
cache = TTLCache(maxsize=200, ttl=900)  # Cache maior (15 min)

# --- Listas de palavras-chave ---
PALAVRAS_CHAVE_ASMA = ["asma", "respiração", "inalador", "bronquite", "alergia", "pulmão", "crise", "sintomas", "tratamento"]
SAUDACOES = ["oi", "olá", "bom dia", "boa tarde", "boa noite", "como vai", "tudo bem"]

# --- Funções de análise ambiental (otimizadas com cache) ---
@lru_cache(maxsize=128)
def analisar_qualidade_ar(aqi):
    niveis = [
        ("Bom", "🟢", "A qualidade do ar é ideal. Não há riscos significativos para pacientes asmáticos."),
        ("Razoável", "🟡", "A qualidade do ar é aceitável. Pacientes sensíveis devem evitar esforços prolongados."),
        ("Moderado", "🟠", "A qualidade do ar pode afetar pacientes com asma. Evite atividades intensas ao ar livre."),
        ("Ruim", "🔴", "A qualidade do ar é ruim. Recomenda-se permanecer em ambientes internos."),
        ("Muito Ruim", "🟣", "A qualidade do ar é muito ruim. Risco elevado à saúde; permaneça em ambientes internos com purificação de ar.")
    ]
    idx = min(max(aqi - 1, 0), 4)
    nivel, emoji, recomendacao = niveis[idx]
    return {"condicao": "AQI", "valor": f"{aqi}", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

@lru_cache(maxsize=128)
def analisar_umidade(humidity):
    if 30 <= humidity <= 50:
        nivel, emoji, recomendacao = "Ideal", "🟢", "Níveis de umidade ideais para pacientes asmáticos."
    elif humidity > 50:
        nivel = "Alta"
        emoji = "🟡" if humidity <= 60 else "🔴"
        recomendacao = "Umidade alta pode agravar sintomas de asma. Considere usar um desumidificador."
    else:
        nivel = "Baixa"
        emoji = "🟡" if humidity >= 20 else "🔴"
        recomendacao = "Umidade baixa pode irritar as vias aéreas. Considere usar um umidificador."
    return {"condicao": "Umidade", "valor": f"{humidity}%", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

@lru_cache(maxsize=128)
def analisar_temperatura(temp):
    if 18 <= temp <= 22:
        nivel, emoji, recomendacao = "Agradável", "🟢", "Temperatura confortável para pacientes asmáticos."
    elif temp < 18:
        nivel = "Fria"
        emoji = "🟡" if temp >= 12 else "🔴"
        recomendacao = "Temperaturas frias podem desencadear broncoespasmo. Proteja-se ao sair."
    else:
        nivel = "Quente"
        emoji = "🟡" if temp <= 28 else "🔴"
        recomendacao = "Calor intenso pode ser um gatilho. Mantenha-se hidratado e evite exposição ao sol."
    return {"condicao": "Temperatura", "valor": f"{temp}°C", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

# --- IA Especialista (RAG otimizado) ---
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Índice '{FAISS_INDEX_PATH}' não encontrado. Execute 'criar_base_de_conhecimento.py'.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
llm = OllamaLLM(model="gemma3", temperature=0.1)

prompt_template = """
Você é um assistente de IA especialista em asma, projetado para fornecer informações precisas e baseadas em evidências cientificas.
Sua missão é responder às perguntas utilizando APENAS as informações contidas no CONTEXTO fornecido abaixo.
Não invente informações nem use conhecimento externo de fontes desconhecidas.
Se a resposta não estiver no contexto, diga: "Desculpe, não encontrei informações sobre isso em minha base de dados. É sempre melhor consultar um profissional de saúde para questões específicas."
Use uma linguagem clara, direta e formal e sempre com o intuito de ensinar/concientisar o paciente.
Se a pergunta não especificar um tema, relacione a resposta à asma.
Para qualquer pergunta sobre sintomas graves, tratamentos ou diagnóstico, finalize com:
"Importante: Esta informação é para fins educativos e não substitui o aconselhamento, diagnóstico ou tratamento médico profissional. Consulte sempre o seu médico para qualquer questão de saúde."

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# --- Função para detectar tipo de pergunta ---
def detectar_tipo_pergunta(pergunta):
    pergunta_lower = pergunta.lower()
    if any(saudacao in pergunta_lower for saudacao in SAUDACOES):
        return "saudacao"
    elif any(palavra in pergunta_lower for palavra in PALAVRAS_CHAVE_ASMA) or pergunta.strip():
        return "asma"
    else:
        return "outro"

# --- Rotas Flask ---
@chat_bp.route("/chat")
def index():
    return render_template("chat.html")

@chat_bp.route("/api/dados-ambientais", methods=["GET"])
async def dados_ambientais():
    try:
        lat, lon, city = await asyncio.to_thread(get_user_location)
        cache_key = f"dados-{lat}-{lon}"
        if cache_key in cache:
            return jsonify(cache[cache_key])

        air_task = asyncio.to_thread(get_air_quality, lat, lon, API_KEY_WEATHER)
        weather_task = asyncio.to_thread(get_weather, lat, lon)
        aqi, pm2_5, pm10 = await air_task
        t, humidity = await weather_task

        temperatura_amb = next((s["valor"] for s in lista_sensores if s["id"] == "temperatura-ambiente"), None)
        sugestoes_list = [
            analisar_qualidade_ar(aqi),
            analisar_temperatura(temperatura_amb if temperatura_amb is not None else t),
            analisar_umidade(humidity)
        ]

        response_data = {"cidade": city, "sugestoes_ambientais": sugestoes_list}
        cache[cache_key] = response_data
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Erro em dados_ambientais: {str(e)}")
        return jsonify({"error": "Não foi possível obter os dados ambientais."}), 500

@chat_bp.route("/api/sugestoes-iniciais", methods=["GET"])
async def sugestoes_iniciais():
    try:
        prompt = (
            "Sugira 3 perguntas curtas e muito comuns que um paciente com asma faria a um especialista. "
            "Responda apenas com as 3 perguntas, cada uma em uma nova linha, sem marcadores."
            "Sempre nas perguntas deixe explícito que se trata sobre um paciente asmático ou uma dúvida com relação à asma."
                 )
        resposta = await asyncio.to_thread(ollama.chat, model="gemma3", messages=[{"role": "user", "content": prompt}])
        sugestoes = [linha.strip() for linha in resposta['message']['content'].split('\n') if linha.strip()][:3]
        return jsonify({"sugestoes": sugestoes})
    except Exception as e:
        logging.error(f"Erro ao gerar sugestões: {str(e)}")
        return jsonify({"error": "Não foi possível gerar sugestões."}), 500

@chat_bp.route("/api/chat", methods=["POST"])
async def api_chat():
    data = request.get_json()
    pergunta = data.get("pergunta", "")
    if not pergunta:
        return jsonify({"error": "A pergunta não pode ser vazia."}), 400

    try:
        tipo_pergunta = detectar_tipo_pergunta(pergunta)
        if tipo_pergunta == "saudacao":
            resposta = "Oi! Como posso ajudar você com relação à asma hoje?"
        elif tipo_pergunta == "asma":
            logging.info(f"Pergunta recebida: '{pergunta}'")
            result = await asyncio.to_thread(qa_chain.invoke, {"query": pergunta})
            resposta = result["result"]
            fontes = set(doc.metadata.get('source', 'Fonte não identificada') for doc in result.get('source_documents', []))
            if fontes:
                resposta += f"\n\n*Fontes consultadas: {', '.join(fontes)}*"
        else:
            resposta = "Desculpe, sou especializado em asma. Posso ajudar com alguma dúvida sobre esse tema?"

        logging.info(f"Resposta enviada: '{resposta}'")
        return jsonify({"resposta": resposta})
    except Exception as e:
        logging.error(f"Erro na rota /api/chat: {str(e)}")
        return jsonify({"error": "Ocorreu um erro inesperado ao processar sua pergunta."}), 500