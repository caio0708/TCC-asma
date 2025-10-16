import requests
from bs4 import BeautifulSoup
import ollama

# Link fixo do artigo sobre asma
URL_ARTIGO_ASMA = "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/asma"

# Função para extrair texto limpo do link
def extrair_conteudo_do_link(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts, styles e seções não informativas
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # Extrai os parágrafos
        texto = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
        return texto.strip()
    except Exception as e:
        raise RuntimeError(f"Erro ao acessar o link: {e}")

# Função para gerar resposta com base no conteúdo
def get_response(pergunta, contexto):
    mensagens = [
        {"role": "system", "content": (
            "Você é um especialista em asma altamente qualificado, com conhecimento baseado no conteúdo do artigo fornecido a seguir e em informações atualizadas da internet. \n"
            "Responda às perguntas do usuário de forma clara, simples e precisa e com a gramática correta, como se estivesse explicando para uma pessoa leiga, sem adicionar informações desnecessárias ou irrelevantes. \n"
            "Foque apenas no que foi perguntado e, se julgar útil, ao final da resposta, sugira uma pergunta relacionada que o usuário possa achar interessante no contexto, começando com 'Você também pode querer saber:'. \n\n"
            + contexto
        )},
        {"role": "user", "content": pergunta}
    ]
    response = ollama.chat(model="llama3.2", messages=mensagens)
    return response['message']['content']

def main():
    print("=== Especialista em Asma (com base em artigo do gov.br) ===")
    print("Carregando conteúdo do site oficial...")

    try:
        contexto = extrair_conteudo_do_link(URL_ARTIGO_ASMA)
        if not contexto:
            raise ValueError("Não foi possível extrair conteúdo útil do link.")
    except Exception as e:
        print(e)
        return

    print("\n✅ Artigo carregado com sucesso!")
    print("Você pode agora fazer perguntas sobre asma com base no artigo do gov.br.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Sua pergunta: ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("Até logo!")
            break

        try:
            resposta = get_response(pergunta, contexto)
            print("\nResposta da IA:\n", resposta)
        except Exception as e:
            print("Erro ao obter resposta:", e)

if __name__ == "__main__":
    main()
