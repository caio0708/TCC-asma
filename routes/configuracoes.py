from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import csv

configuracoes_bp = Blueprint('configuracoes', __name__)

# Diretório e arquivo de usuários
DATA_DIR = os.path.join(os.getcwd(), 'dados')
USUARIOS_CSV = os.path.join(DATA_DIR, 'usuarios.csv')

# Garante existência da pasta de dados
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

@configuracoes_bp.route('/configuracoes', methods=['GET', 'POST'])
def configuracoes():
    if request.method == 'POST':
        dados, erro = validar_formulario(request.form)
        if erro:
            flash(erro, 'danger')
            return redirect(url_for('configuracoes.configuracoes'))

        username, age, gender, altura = dados
        salvar_usuario_csv(username, age, gender, altura)

        # Atualiza sessão do usuário
        session['username'] = username
        session['age']      = age
        session['gender']   = gender
        session['altura']   = altura

        flash(f"Dados salvos com sucesso! Bem-vindo, {username}.", "success")
        return redirect(url_for('configuracoes.configuracoes'))

    # GET: carrega dados da sessão para o formulário
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