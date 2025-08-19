import os
from dotenv import load_dotenv
import logging
import asyncio
import traceback
from functools import lru_cache
from operator import itemgetter

from flask import Blueprint, render_template, request, jsonify
from cachetools import TTLCache
import ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from routes.api import get_user_location, get_air_quality, get_weather
from routes.sensores import lista_sensores

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FAISS_INDEX_PATH = "faiss_index"
load_dotenv()

chat_bp = Blueprint('chat', __name__)
cache = TTLCache(maxsize=200, ttl=900)

# --- Listas de palavras-chave ---
PALAVRAS_CHAVE_ASMA = ["asma", "respiração", "inalador", "bronquite", "alergia", "pulmão", "crise", "sintomas", "tratamento", "como estou", "meus dados", "analise meus sensores"]
SAUDACOES = ["oi", "olá", "bom dia", "boa tarde", "boa noite", "como vai", "tudo bem","ola"]

# --- Funções de análise ambiental (sem alterações) ---
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
        nivel, emoji, recomendacao = "Alta", "🟡" if humidity <= 60 else "🔴", "Umidade alta pode agravar sintomas de asma. Considere usar um desumidificador."
    else:
        nivel, emoji, recomendacao = "Baixa", "🟡" if humidity >= 20 else "🔴", "Umidade baixa pode irritar as vias aéreas. Considere usar um umidificador."
    return {"condicao": "Umidade", "valor": f"{humidity}%", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

@lru_cache(maxsize=128)
def analisar_temperatura(temp):
    if 18 <= temp <= 22:
        nivel, emoji, recomendacao = "Agradável", "🟢", "Temperatura confortável para pacientes asmáticos."
    elif temp < 18:
        nivel, emoji, recomendacao = "Fria", "🟡" if temp >= 12 else "🔴", "Temperaturas frias podem desencadear broncoespasmo. Proteja-se ao sair."
    else:
        nivel, emoji, recomendacao = "Quente", "🟡" if temp <= 28 else "🔴", "Calor intenso pode ser um gatilho. Mantenha-se hidratado e evite exposição ao sol."
    return {"condicao": "Temperatura", "valor": f"{temp}°C", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

# --- IA Especialista ---
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Índice '{FAISS_INDEX_PATH}' não encontrado. Execute 'fontes.py'.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
llm = OllamaLLM(model="gemma3", temperature=0.1)

prompt_template = """
Você é um assistente de IA especialista em asma, atuando como um analista de saúde.
Sua missão é interpretar os DADOS DOS SENSORES DO PACIENTE, usando o CONTEXTO CIENTÍFICO como referência para os valores normais e de risco.
Com base nessa análise, responda à PERGUNTA DO USUÁRIO, fornecendo um feedback claro sobre o seu estado respiratório atual.
Se a pergunta for "como estou?", faça uma análise completa de todos os sensores. Destaque quaisquer valores que estejam fora do ideal e explique o que isso significa.
Seja claro, objetivo e empático.

**Exemplo de Análise:**
- Se a saturação de oxigênio for 94%, você deve identificar que está abaixo do ideal (95-100%) e explicar o risco de hipoxemia.
- Se a frequência respiratória for 22 ipm, aponte que está elevada para um adulto em repouso (normal: 12-20 ipm) e que isso pode indicar dificuldade respiratória.

Use apenas o CONTEXTO CIENTÍFICO. Não invente informações.
Finalize TODAS as respostas de análise com o aviso: "Importante: Esta é uma análise para fins educativos e não substitui um diagnóstico médico. Consulte seu médico."

CONTEXTO CIENTÍFICO:
{context}

DADOS DOS SENSORES DO PACIENTE:
{sensor_data}

PERGUNTA DO USUÁRIO:
{question}

ANÁLISE E RESPOSTA:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "sensor_data"],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnableParallel(
        context=(itemgetter("question") | retriever | format_docs),
        question=itemgetter("question"),
        sensor_data=itemgetter("sensor_data"),
        source_documents=(itemgetter("question") | retriever)
    )
    | {
        "result": PROMPT | llm | StrOutputParser(),
        "source_documents": itemgetter("source_documents")
    }
)

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
        if cache_key in cache: return jsonify(cache[cache_key])
        api_key = os.getenv('API_WEATHER_KEY')
        air_task = asyncio.to_thread(get_air_quality, lat, lon, api_key)
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
            "Sugira 3 perguntas curtas e muito comuns que um paciente com asma faria a um especialista. Inclua sempre 1 dessas perguntas algo como 'Como estou agora?' ou 'Meus dados estão normais?'. "
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
            resposta = "Olá! Sou seu assistente para monitoramento da asma. Como posso ajudar a analisar seus dados hoje?"
        elif tipo_pergunta == "asma":
            logging.info(f"Pergunta recebida: '{pergunta}'")
            
            # ✅ CORREÇÃO: Trocado s['nome'] por s['id'] para corresponder à estrutura de dados.
            # Adicionado .replace('-', ' ').title() para formatar o nome para exibição.
            # Ex: "frequencia-respiratoria" se torna "Frequencia Respiratoria".
            sensor_data_str = "\n".join([f"- {s['id'].replace('-', ' ').title()}: {s['valor']} {s['unidade']}" for s in lista_sensores])
            if not sensor_data_str:
                sensor_data_str = "Dados dos sensores não disponíveis."
            
            logging.info(f"Dados dos sensores formatados:\n{sensor_data_str}")

            result = await asyncio.to_thread(
                chain.invoke,
                {"question": pergunta, "sensor_data": sensor_data_str}
            )
            
            resposta = result["result"]
            fontes = set(doc.metadata.get('source', 'Fonte não identificada') for doc in result.get('source_documents', []))
            if fontes:
                resposta += f"\n\n*Fontes consultadas: {', '.join(fontes)}*"
        else:
            resposta = "Desculpe, sou um assistente especializado em asma. Você tem alguma dúvida sobre o tema?"

        logging.info(f"Resposta enviada: '{resposta}'")
        return jsonify({"resposta": resposta})
    except Exception as e:
        logging.error(f"Erro na rota /api/chat: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Ocorreu um erro inesperado ao processar sua pergunta."}), 500