# fontes.py — VERSÃO CORRIGIDA

import os
import logging
from dotenv import load_dotenv  # <-- 1. IMPORTAR A BIBLIOTECA

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Carregar variáveis de ambiente do arquivo .env ---
load_dotenv() # <-- 2. CHAMAR A FUNÇÃO NO INÍCIO DO SCRIPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URLS_ARTIGOS_ASMA = [
    "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/asma",
    "https://www.who.int/news-room/fact-sheets/detail/asthma",
    "https://www.tuasaude.com/asma/",
    "http://scielo.iec.gov.br/scielo.php?script=sci_arttext&pid=S1982-32582007000100007",
    "https://bvsms.saude.gov.br/?p=482",
    "https://www.msdmanuals.com/pt/profissional/dist%C3%BArbios-pulmonares/asma-e-doen%C3%A7as-relacionadas/asma?query=asma#Fisiopatologia_v913580_pt",
    "https://abrasaopaulo.org/perguntas-frequentes/",
]

FAISS_INDEX_PATH = "faiss_index"

def criar_e_salvar_base_de_conhecimento():
    logging.info("Carregando documentos das URLs...")
    loader = WebBaseLoader(URLS_ARTIGOS_ASMA)
    docs = loader.load()
    logging.info(f"{len(docs)} documentos carregados.")

    # Chunking p/ RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)
    logging.info(f"Documentos divididos em {len(splits)} chunks.")

    logging.info("Criando embeddings (Gemini) e indexando no FAISS...")

    # Esta função agora vai encontrar a variável de ambiente corretamente
    def _embed_model_name():
        name = os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip()
        return name if name.startswith("models/") else f"models/{name}"

    embeddings = GoogleGenerativeAIEmbeddings(
        model=_embed_model_name(),
        # A chave de API agora será encontrada por os.environ
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    logging.info(f"Base de conhecimento salva com sucesso em '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    criar_e_salvar_base_de_conhecimento()