# chat.py — VERSÃO GEMINI

import os
from dotenv import load_dotenv
import logging
import asyncio
import traceback
from functools import lru_cache
from operator import itemgetter

from flask import Blueprint, render_template, request, jsonify
from cachetools import TTLCache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# >>> TROCA: Azure → Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from routes.api import get_user_location, get_air_quality, get_weather
from routes.sensores import lista_sensores

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index')
load_dotenv()

chat_bp = Blueprint('chat', __name__)
cache = TTLCache(maxsize=200, ttl=900)

PALAVRAS_CHAVE_ASMA = ["asma", "respiração", "inalador", "bronquite", "alergia", "pulmão", "crise", "sintomas", "tratamento", "como estou", "meus dados", "analise meus sensores"]
SAUDACOES = ["oi", "olá", "bom dia", "boa tarde", "boa noite", "como vai", "tudo bem","ola"]

# --- Análises ambientais  ---
@lru_cache(maxsize=128)
def analisar_qualidade_ar(aqi):
    niveis = [
        ("Bom", "🟢", "A qualidade do ar é ideal. Não há riscos significativos para pacientes asmáticos."),
        ("Razoável", "🟡", "A qualidade do ar é aceitável. Pacientes sensíveis devem evitar esforços prolongados."),
        ("Moderado", "🟠", "A qualidade do ar pode afetar pacientes com asma. Evite atividades intensas ao ar livre."),
        ("Ruim", "🔴", "A qualidade do ar é ruim. Recomenda-se permanecer em ambientes internos."),
        ("Muito Ruim", "🟣", "Risco elevado à saúde; permaneça em ambientes internos com purificação de ar."),
    ]
    idx = min(max(aqi - 1, 0), 4)
    nivel, emoji, recomendacao = niveis[idx]
    return {"condicao": "AQI", "valor": f"{aqi}", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

@lru_cache(maxsize=128)
def analisar_umidade(humidity):
    if 30 <= humidity <= 50:
        nivel, emoji, recomendacao = "Ideal", "🟢", "Níveis de umidade ideais para pacientes asmáticos."
    elif humidity > 50:
        nivel, emoji, recomendacao = "Alta", "🟡" if humidity <= 60 else "🔴", "Umidade alta pode agravar sintomas; considere desumidificador."
    else:
        nivel, emoji, recomendacao = "Baixa", "🟡" if humidity >= 20 else "🔴", "Umidade baixa pode irritar vias aéreas; considere umidificador."
    return {"condicao": "Umidade", "valor": f"{humidity}%", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}

@lru_cache(maxsize=128)
def analisar_temperatura(temp):
    if 18 <= temp <= 22:
        nivel, emoji, recomendacao = "Agradável", "🟢", "Temperatura confortável para pacientes asmáticos."
    elif temp < 18:
        nivel, emoji, recomendacao = "Fria", "🟡" if temp >= 12 else "🔴", "Frio pode desencadear broncoespasmo; proteja-se ao sair."
    else:
        nivel, emoji, recomendacao = "Quente", "🟡" if temp <= 28 else "🔴", "Calor intenso pode ser gatilho; hidrate-se e evite sol."
    return {"condicao": "Temperatura", "valor": f"{temp}°C", "nivel": nivel, "emoji": emoji, "recomendacao": recomendacao}


# --- IA Especialista (RAG) ---
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Índice '{FAISS_INDEX_PATH}' não encontrado. Execute 'fontes.py' para criá-lo.")

# --- helper para normalizar o nome do modelo de embeddings
def _embed_model_name():
    name = os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip()
    return name if name.startswith("models/") else f"models/{name}"

embeddings = GoogleGenerativeAIEmbeddings(
    model=_embed_model_name(),
    google_api_key=os.environ["GOOGLE_API_KEY"],
)


vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# >>> LLM principal (Gemini) — flash = mais rápido
llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# --- Prompt especialista em asma (mantido com ajustes mínimos) ---
prompt_template = """
Você é um assistente de IA especialista em asma, em PT-BR, empático e baseado em evidências.

=== Regras Fundamentais ===
1) Foco na intenção do usuário:
   - Se a pergunta for GERAL sobre asma (ex.: causas, prevenção, sintomas, tratamentos, informações de referência), responda apenas com base em evidências científicas, sem usar dados de sensores.
   - Se a pergunta for PESSOAL sobre o estado atual (ex.: "como estou?", "meus dados estão normais?", "analise meus sensores"), sua prioridade MÁXIMA é analisar os `DADOS ATUAIS DOS SENSORES DO PACIENTE`.

2) Processo de análise personalizada (somente quando aplicável):
   a. Para cada sensor relevante, busque a faixa de normalidade usando a ferramenta `resposta_com_grounding`.
   b. Compare o valor do paciente com a faixa de referência.
   c. Explique o que isso significa para uma pessoa com asma, destacando riscos quando valores estiverem fora do ideal.

3) Regra de citação:
   - Ao usar a ferramenta `resposta_com_grounding`, cite a fonte SOMENTE se houver link confiável disponível.
   - Use o formato: [fonte: exemplo.com]
   - Se não houver fonte, NÃO invente citações nem placeholders. Simplesmente não cite.

4) Aviso médico:
   - Qualquer resposta que envolva sensores, sintomas ou sugestões deve terminar com:
   "Importante: Esta é uma análise para fins educativos e não substitui um diagnóstico ou aconselhamento médico profissional. Se você não está se sentindo bem, consulte seu médico."

5) Linguagem:
   - Sempre em português (PT-BR).
   - Empática, clara, objetiva e acessível. Evite jargões médicos complexos.

=== Instruções de Resposta ===
- Se PERGUNTA GERAL: responda de forma científica e direta, sem personalização.
- Se PERGUNTA SOBRE ESTADO ATUAL: analise os sensores com base no CONTEXTO CIENTÍFICO e referências externas (grounding).
- Não invente dados. Não force análise personalizada quando a intenção não for essa.
- Nunca invente links. Só cite se houver fonte real e confiável.

=== Contexto fornecido ===
CONTEXTO CIENTÍFICO:
{context}

DADOS ATUAIS DOS SENSORES DO PACIENTE:
{sensor_data}

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA:
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
    if any(s in pergunta_lower for s in SAUDACOES):
        return "saudacao"
    elif any(p in pergunta_lower for p in PALAVRAS_CHAVE_ASMA) or pergunta.strip():
        return "asma"
    else:
        return "outro"

# --- Função utilitária: Grounding com Google Search para perguntas gerais ---
async def resposta_com_grounding(pergunta: str) -> str:
    """
    Usa o Gemini com Grounding (Pesquisa Google) para perguntas não diretamente ligadas aos sensores,
    retornando conteúdo com fontes/citações.
    """

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)

    # Ativa Grounding com Google Search (docs oficiais abaixo)
    # https://ai.google.dev/gemini-api/docs/google-search
    # https://ai.google.dev/gemini-api/docs/grounding
    tools = {"google_search_retrieval": {}}

    # Recomendado: pedir citações inline
    response = await asyncio.to_thread(
        model.generate_content,
        contents=[{"role": "user", "parts": [pergunta]}],
        tools=tools,
        generation_config={"temperature": 0.2, "top_p": 0.9},
        # safety_settings pode ser ajustado aqui também
    )

    text = getattr(response, "text", None) or "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
    # Opcional: anexar fontes quando disponíveis
    try:
        citations = []
        grounding_metadata = response.candidates[0].grounding_metadata
        if grounding_metadata and getattr(grounding_metadata, "grounding_chunks", None):
            for chunk in grounding_metadata.grounding_chunks:
                if getattr(chunk, "web", None) and getattr(chunk.web, "uri", None):
                    citations.append(chunk.web.uri)
        if citations:
            text += "\n\nFontes: " + ", ".join(citations[:5])
    except Exception:
        pass

    return text or "Não encontrei resultados suficientes no momento."

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

        api_key = os.getenv('API_WEATHER_KEY')
        air_task = asyncio.to_thread(get_air_quality, lat, lon, api_key)
        weather_task = asyncio.to_thread(get_weather, lat, lon)
        aqi, pm2_5, pm10, o3, no2, so2 = await air_task
        t, humidity = await weather_task

        temperatura_amb = next((s["valor"] for s in lista_sensores if s["id"] == "temperatura-ambiente"), None)
        sugestoes_list = [
            analisar_qualidade_ar(aqi),
            analisar_temperatura(temperatura_amb if temperatura_amb is not None else t),
            analisar_umidade(humidity),
        ]
        response_data = {"cidade": city, "sugestoes_ambientais": sugestoes_list}
        cache[cache_key] = response_data
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Erro em dados_ambientais: {str(e)}")
        return jsonify({"error": "Não foi possível obter os dados ambientais."}), 500

@chat_bp.route("/api/sugestoes-iniciais", methods=["GET"])
def sugestoes_iniciais():
    """
    Retorna uma lista fixa de perguntas sugeridas para o chat.
    """
    try:
        # Lista de perguntas fixas, conforme solicitado
        sugestoes = [
            "Como estão meus sintomas?",
            "Faça um relatório da minha situação atual",
            "Como está o ambiente para um asmático?"
        ]
        return jsonify({"sugestoes": sugestoes})
    except Exception as e:
        # É bom manter o log de erros, caso algo inesperado aconteça
        logging.error(f"Erro inesperado ao gerar sugestões fixas: {str(e)}")
        return jsonify({"error": "Não foi possível carregar as sugestões."}), 500


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

            sensor_data_str = "\n".join(
                [f"- {s['id'].replace('-', ' ').title()}: {s['valor']} {s['unidade']}" for s in lista_sensores]
            ) or "Dados dos sensores não disponíveis."
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
            # Para perguntas gerais: usar Grounding com Google Search
            resposta = await resposta_com_grounding(pergunta)

        logging.info(f"Resposta enviada: '{resposta}'")
        return jsonify({"resposta": resposta})
    except Exception as e:
        logging.error(f"Erro na rota /api/chat: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Ocorreu um erro inesperado ao processar sua pergunta."}), 500
