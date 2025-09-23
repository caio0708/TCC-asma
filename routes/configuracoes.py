from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import csv
from datetime import datetime
import pandas as pd

configuracoes_bp = Blueprint('configuracoes', __name__)

# --- CONSTANTES DE ARQUIVOS E DIRETÓRIOS ---
DATA_DIR = os.path.join(os.getcwd(), 'dados')
USUARIOS_CSV = os.path.join(DATA_DIR, 'usuarios.csv')
# Novo arquivo para o registro de crises
REGISTRO_CRISE_CSV = os.path.join(DATA_DIR, 'registro_crise.csv')

# Garante a existência da pasta de dados
os.makedirs(DATA_DIR, exist_ok=True)

def salvar_usuario_csv(username, age, gender, altura):
    """Salva dados do usuário no CSV."""
    file_exists = os.path.isfile(USUARIOS_CSV)
    with open(USUARIOS_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Username", "Age", "Gender", "Altura"])
        writer.writerow([username, age, gender, altura])

def validar_formulario(form):
    """Valida dados do formulário e retorna tupla (dados, mensagem_erro)."""
    username = form.get('username', '').strip()
    age_str  = form.get('age', '').strip()
    gender   = form.get('gender', '').strip()
    altura   = form.get('altura', '').strip()

    if not username or not age_str or not gender:
        return None, 'Todos os campos obrigatórios devem ser preenchidos.'

    try:
        age = int(age_str)
        if age <= 0:
            raise ValueError
    except ValueError:
        return None, 'Idade deve ser um número inteiro positivo.'

    return (username, age, gender, altura), None

def atualizar_ultima_crise_usuario(username, timestamp):
    """
    Lê o usuarios.csv, adiciona/atualiza a coluna 'UltimaCrise'
    para um usuário específico e salva o arquivo.
    """
    try:
        df = pd.read_csv(USUARIOS_CSV)

        if 'UltimaCrise' not in df.columns:
            df['UltimaCrise'] = ''

        df.loc[df['Username'] == username, 'UltimaCrise'] = timestamp
        df.to_csv(USUARIOS_CSV, index=False, encoding='utf-8')
        return True
    except FileNotFoundError:
        print(f"Arquivo {USUARIOS_CSV} não encontrado.")
        return False
    except Exception as e:
        print(f"Ocorreu um erro ao atualizar o CSV: {e}")
        return False

@configuracoes_bp.route('/configuracoes', methods=['GET', 'POST'])
def configuracoes():
    if request.method == 'POST':
        dados, erro = validar_formulario(request.form)
        if erro:
            flash(erro, 'danger')
            return redirect(url_for('configuracoes.configuracoes'))

        username, age, gender, altura = dados
        salvar_usuario_csv(username, age, gender, altura)

        session['username'] = username
        session['age']      = age
        session['gender']   = gender
        session['altura']   = altura

        flash(f"Dados salvos com sucesso! Bem-vindo, {username}.", "success")
        return redirect(url_for('configuracoes.configuracoes'))

    return render_template(
        'configuracoes.html',
        username=session.get('username', ''),
        age=session.get('age', ''),
        gender=session.get('gender', ''),
        altura=session.get('altura', '')
    )

@configuracoes_bp.route('/sair')
def logout():
    session.clear()
    flash('Você saiu da sessão.', 'info')
    return redirect(url_for('configuracoes.configuracoes'))

# --- ROTA PARA REGISTRAR A CRISE (ATUALIZADA) ---
@configuracoes_bp.route('/registrar_crise', methods=['POST'])
def registrar_crise():
    """
    Registra um evento de crise para o usuário logado, salvando o histórico
    em registro_crise.csv com contagem e atualizando o status em usuarios.csv.
    """
    if 'username' not in session:
        flash('Você precisa estar logado para registrar uma crise.', 'warning')
        return redirect(url_for('configuracoes.configuracoes'))
        
    username = session['username']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- PARTE 1: Salva o registro em registro_crise.csv com contagem ----
    contagem = 1
    try:
        # Se o arquivo já existir, lê para contar as crises anteriores do usuário
        if os.path.isfile(REGISTRO_CRISE_CSV):
            df_crises = pd.read_csv(REGISTRO_CRISE_CSV)
            # Filtra pelo usuário atual e conta quantas vezes ele já aparece
            contagem_anterior = df_crises[df_crises['Username'] == username].shape[0]
            contagem = contagem_anterior + 1

        # Adiciona a nova linha de crise ao arquivo
        file_exists = os.path.isfile(REGISTRO_CRISE_CSV)
        with open(REGISTRO_CRISE_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escreve o cabeçalho se o arquivo for novo
            if not file_exists:
                writer.writerow(["Username", "Timestamp", "Contagem"])
            writer.writerow([username, timestamp, contagem])

    except Exception as e:
        flash(f'Ocorreu um erro ao salvar o registro da crise: {e}', 'danger')
        return redirect(url_for('configuracoes.configuracoes'))
        
    # ---- PARTE 2: Atualiza a coluna 'UltimaCrise' no usuarios.csv ----
    sucesso_update = atualizar_ultima_crise_usuario(username, timestamp)

    if sucesso_update:
        flash(f'Crise registrada para {username} em {timestamp}. Esta é a sua {contagem}ª crise registrada.', 'success')
    else:
        # Informa que o registro principal foi salvo, mas o perfil não foi atualizado
        flash(f'Crise registrada no histórico, mas falha ao atualizar o perfil do usuário.', 'warning')
        
    return redirect(url_for('configuracoes.configuracoes'))