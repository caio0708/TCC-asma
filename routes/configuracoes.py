from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
import csv
from datetime import datetime
import pandas as pd
# 1. IMPORTADO PARA SEGURANÇA DE SENHA
from werkzeug.security import generate_password_hash, check_password_hash

configuracoes_bp = Blueprint('configuracoes', __name__)

# --- CONSTANTES DE ARQUIVOS E DIRETÓRIOS ---
DATA_DIR = os.path.join(os.getcwd(), 'dados')
USUARIOS_CSV = os.path.join(DATA_DIR, 'usuarios.csv')
REGISTRO_CRISE_CSV = os.path.join(DATA_DIR, 'registro_crise.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. FUNÇÃO 'salvar_usuario_csv' ATUALIZADA ---
def salvar_usuario_csv(username, password, age, gender, altura):
    """Salva dados do usuário (com senha hasheada) no CSV."""
    
    # Gera o hash seguro da senha
    password_hash = generate_password_hash(password)
    
    file_exists = os.path.isfile(USUARIOS_CSV)
    with open(USUARIOS_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Adiciona a coluna 'Password' ao cabeçalho
            writer.writerow(["Username", "Password", "Age", "Gender", "Altura"])
        # Salva o hash, não a senha original
        writer.writerow([username, password_hash, age, gender, altura])


def validar_formulario(form):
    """Valida dados do formulário e retorna tupla (dados, mensagem_erro)."""
    username = form.get('username', '').strip()
    age_str  = form.get('age', '').strip()
    gender   = form.get('gender', '').strip()
    altura_str = form.get('altura', '').strip()

    if not all([username, age_str, gender, altura_str]):
        return None, 'Todos os campos são obrigatórios.'
    
    try:
        age = int(age_str)
        altura = float(altura_str)
        if age <= 0 or altura <= 0:
            raise ValueError
    except ValueError:
        return None, 'Idade e altura devem ser números positivos.'
        
    return (username, age, gender, altura), None

def atualizar_dados_usuario(username, age, gender, altura):
    """Atualiza os dados de um usuário existente no CSV."""
    if not os.path.isfile(USUARIOS_CSV):
        flash('Arquivo de usuários não encontrado.', 'danger')
        return False
        
    try:
        df = pd.read_csv(USUARIOS_CSV)
        
        user_index = df.index[df['Username'] == username].tolist()
        if not user_index:
            return False 

        df.loc[df['Username'] == username, ['Age', 'Gender', 'Altura']] = [age, gender, altura]
        
        df.to_csv(USUARIOS_CSV, index=False)
        return True

    except Exception as e:
        print(f"Erro ao atualizar CSV: {e}")
        return False

def atualizar_ultima_crise_usuario(username, timestamp):
    """Atualiza a coluna 'UltimaCrise' para um usuário específico."""
    try:
        df = pd.read_csv(USUARIOS_CSV)
        
        if 'UltimaCrise' not in df.columns:
            df['UltimaCrise'] = None 
        
        df.loc[df['Username'] == username, 'UltimaCrise'] = timestamp
        df.to_csv(USUARIOS_CSV, index=False)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Erro ao atualizar a última crise no CSV: {e}")
        return False

# --- ROTA PRINCIPAL (LOGIN + CONFIGURAÇÕES) ---
@configuracoes_bp.route('/configuracoes', methods=['GET', 'POST'])
def configuracoes():
    
    if request.method == 'POST':
        
        # --- LÓGICA DE LOGIN ---
        if 'username_login' in request.form:
            username = request.form.get('username_login')
            password = request.form.get('password_login') 

            try:
                df_usuarios = pd.read_csv(USUARIOS_CSV)
                usuario_encontrado = df_usuarios[df_usuarios['Username'] == username]
                
                # --- 3. LÓGICA DE LOGIN CORRIGIDA ---
                # Verifica se o usuário existe E se a senha está correta
                if not usuario_encontrado.empty and check_password_hash(usuario_encontrado.iloc[0]['Password'], password):
                    
                    # Coloca os dados do usuário na sessão
                    session['username'] = usuario_encontrado.iloc[0]['Username']
                    # Converte para int/float nativos do Python para segurança da sessão
                    session['age'] = int(usuario_encontrado.iloc[0]['Age'])
                    session['gender'] = usuario_encontrado.iloc[0]['Gender']
                    session['altura'] = float(usuario_encontrado.iloc[0]['Altura'])
                    
                    flash('Login bem-sucedido!', 'success')
                    return redirect(url_for('configuracoes.configuracoes'))
                else:
                    # Esta mensagem agora está correta (usuário não existe OU senha errada)
                    flash('Usuário ou senha inválidos.', 'danger')
                    return redirect(url_for('configuracoes.configuracoes'))

            except FileNotFoundError:
                flash('Arquivo de usuários não encontrado. Crie uma conta primeiro.', 'warning')
                return redirect(url_for('configuracoes.configuracoes'))
            except KeyError:
                flash('Erro de login. O arquivo de usuários pode estar corrompido ou sem a coluna "Password".', 'danger')
                return redirect(url_for('configuracoes.configuracoes'))
            except Exception as e:
                flash(f'Erro ao fazer login: {e}', 'danger')
                return redirect(url_for('configuracoes.configuracoes'))

        # --- LÓGICA DE ATUALIZAÇÃO DE CONFIGURAÇÕES ---
        elif 'age' in request.form:
            if 'username' not in session:
                flash('Sessão expirada. Faça login novamente.', 'warning')
                return redirect(url_for('configuracoes.configuracoes'))
            
            username_sessao = session['username']
            age_str = request.form.get('age', '').strip()
            gender = request.form.get('gender', '').strip()
            altura_str = request.form.get('altura', '').strip()

            if not all([age_str, gender, altura_str]):
                flash('Todos os campos são obrigatórios.', 'danger')
            else:
                try:
                    age = int(age_str)
                    altura = float(altura_str)
                    if age <= 0 or altura <= 0:
                        raise ValueError
                    
                    if atualizar_dados_usuario(username_sessao, age, gender, altura):
                        session['age'] = age
                        session['gender'] = gender
                        session['altura'] = altura
                        flash('Dados atualizados com sucesso!', 'success')
                    else:
                        flash('Erro ao atualizar os dados.', 'danger')

                except ValueError:
                    flash('Idade e altura devem ser números positivos.', 'danger')
            
            return redirect(url_for('configuracoes.configuracoes'))

    # --- LÓGICA GET ---
    return render_template('configuracoes.html')

# --- 4. NOVA ROTA DE REGISTRO (CORRIGE O BUILDERROR) ---
@configuracoes_bp.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        age_str  = request.form.get('age', '').strip()
        gender   = request.form.get('gender', '').strip()
        altura_str = request.form.get('altura', '').strip()

        # Validação
        if not all([username, password, age_str, gender, altura_str]):
            flash('Todos os campos são obrigatórios.', 'danger')
            return redirect(url_for('configuracoes.registro'))
        
        try:
            age = int(age_str)
            altura = float(altura_str)
        except ValueError:
            flash('Idade e altura devem ser números.', 'danger')
            return redirect(url_for('configuracoes.registro'))

        # Verifica se usuário já existe
        try:
            if os.path.isfile(USUARIOS_CSV):
                df_usuarios = pd.read_csv(USUARIOS_CSV)
                if not df_usuarios[df_usuarios['Username'] == username].empty:
                    flash('Este nome de usuário já está em uso.', 'danger')
                    return redirect(url_for('configuracoes.registro'))
        except Exception as e:
            flash(f'Erro ao ler arquivo de usuários: {e}', 'danger')
            return redirect(url_for('configuracoes.registro'))

        # Salva o novo usuário
        try:
            salvar_usuario_csv(username, password, age, gender, altura)
            flash('Conta criada com sucesso! Faça o login.', 'success')
            return redirect(url_for('configuracoes.configuracoes')) # Manda para o login
        except Exception as e:
            flash(f'Erro ao salvar usuário: {e}', 'danger')
            return redirect(url_for('configuracoes.registro'))

    # Se for GET, apenas mostra a página de registro
    return render_template('registro.html')


@configuracoes_bp.route('/logout')
def logout():
    session.clear()
    flash('Você saiu da sua conta.', 'info')
    return redirect(url_for('configuracoes.configuracoes'))


@configuracoes_bp.route('/registrar_crise', methods=['POST'])
def registrar_crise():
    if 'username' not in session:
        flash('Faça login para registrar uma crise.', 'warning')
        return redirect(url_for('configuracoes.configuracoes'))

    username = session['username']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    contagem = 1

    try:
        if os.path.isfile(REGISTRO_CRISE_CSV):
            df_crises = pd.read_csv(REGISTRO_CRISE_CSV)
            contagem_anterior = df_crises[df_crises['Username'] == username].shape[0]
            contagem = contagem_anterior + 1

        file_exists = os.path.isfile(REGISTRO_CRISE_CSV)
        with open(REGISTRO_CRISE_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Username", "Timestamp", "Contagem"])
            writer.writerow([username, timestamp, contagem])

    except Exception as e:
        flash(f'Ocorreu um erro ao salvar o registro da crise: {e}', 'danger')
        return redirect(url_for('configuracoes.configuracoes'))
        
    sucesso_update = atualizar_ultima_crise_usuario(username, timestamp)

    if sucesso_update:
        flash(f'Crise registrada para {username} em {timestamp}. Esta é a sua {contagem}ª crise registrada.', 'success')
    else:
        flash(f'Crise registrada em {timestamp}, mas houve um erro ao atualizar o seu perfil.', 'warning')

    return redirect(url_for('configuracoes.configuracoes'))