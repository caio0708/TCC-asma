import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URLS_ARTIGOS_ASMA = [
    "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/asma",
    "https://www.who.int/news-room/fact-sheets/detail/asthma",
    "https://www.tuasaude.com/asma/",
    "http://scielo.iec.gov.br/scielo.php?script=sci_arttext&pid=S1982-32582007000100007",
   # "https://ginasthma.org/wp-content/uploads/2022/07/GINA-Main-Report-2022-FINAL-22-07-01-WMS.pdf",
    "https://bvsms.saude.gov.br/?p=482",
    "https://www.msdmanuals.com/pt/profissional/dist%C3%BArbios-pulmonares/asma-e-doen%C3%A7as-relacionadas/asma?query=asma#Fisiopatologia_v913580_pt",
    "https://abrasaopaulo.org/perguntas-frequentes/",
]

FAISS_INDEX_PATH = "faiss_index"

def criar_e_salvar_base_de_conhecimento():
    logging.info("Iniciando o carregamento dos documentos das URLs...")
    loader = WebBaseLoader(URLS_ARTIGOS_ASMA)
    docs = loader.load()
    logging.info(f"{len(docs)} documentos carregados com sucesso.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    logging.info(f"Documentos divididos em {len(splits)} pedaços (chunks).")

    logging.info("Criando embeddings e indexando no FAISS. Isso pode levar alguns minutos...")

    # ❗ USE UM MODELO COMPATÍVEL COM EMBEDDINGS
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    logging.info(f"Base de conhecimento salva com sucesso em '{FAISS_INDEX_PATH}'")

if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH):
        criar_e_salvar_base_de_conhecimento()
    else:
        logging.info(f"O diretório de índice '{FAISS_INDEX_PATH}' já existe. Nenhuma ação foi tomada.")
