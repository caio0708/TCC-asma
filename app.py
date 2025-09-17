from flask import Flask
from routes.painel import painel_bp
from routes.sensores import sensores_bp
from routes.chat import chat_bp
from routes.insights import insights_bp
from routes.configuracoes import configuracoes_bp

app = Flask(__name__)

app.secret_key = 'uma-senha-muito-secreta-e-complexa'

# Registro de blueprints
app.register_blueprint(sensores_bp)
app.register_blueprint(painel_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(insights_bp)
app.register_blueprint(configuracoes_bp)

if __name__ == '__main__':
    app.run(debug=True)

