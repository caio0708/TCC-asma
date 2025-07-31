from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import csv

configuracoes_bp = Blueprint('configuracoes', __name__)

# Define diretório e caminho do CSV
DATA_DIR = os.path.join(os.getcwd(), 'dados')
USUARIOS_CSV = os.path.join(DATA_DIR, 'usuarios.csv')

# Garante existência da pasta de dados
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

@configuracoes_bp.route('/configuracoes', methods=['GET', 'POST'])
def configuracoes():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        age_str  = request.form.get('age', '').strip()
        gender   = request.form.get('gender', '').strip()
        altura   = request.form.get('altura', '').strip()

        if not username or not age_str or not gender:
            flash('Todos os campos são obrigatórios.', 'danger')
            return redirect(url_for('configuracoes.configuracoes'))

        try:
            age = int(age_str)
        except ValueError:
            flash('Idade deve ser um número inteiro.', 'danger')
            return redirect(url_for('configuracoes.configuracoes'))

        # Salva no CSV
        file_exists = os.path.isfile(USUARIOS_CSV)
        with open(USUARIOS_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Username", "Age", "Gender","Altura"])
            writer.writerow([username, age, gender, altura])

        # Sobrescreve a sessão atual
        session['username'] = username
        session['age']      = age
        session['gender']   = gender
        session['altura']   = altura

        flash(f"Dados salvos com sucesso! Bem-vindo, {username}.", "success")
        return redirect(url_for('configuracoes.configuracoes'))

    # GET
    username = session.get('username', '')
    age      = session.get('age', '')
    gender   = session.get('gender', '')
    altura   = session.get('altura', '')
    return render_template('configuracoes.html', username=username, age=age, gender=gender, altura=altura)


@configuracoes_bp.route('/sair')
def logout():
    session.clear()
    flash('Você saiu da sessão.', 'info')
    return redirect(url_for('configuracoes.configuracoes'))
