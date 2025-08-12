
// Funções enviarPergunta, adicionarMensagem, removerMensagem (sem alterações)
async function enviarPergunta(pergunta) {
  if (!pergunta.trim()) return;
  adicionarMensagem("user", pergunta);
  const loadingMessageId = adicionarMensagem("bot", "Assistente IA está digitando...");
  document.getElementById("pergunta-input").value = "";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta })
    });
    const data = await response.json();
    removerMensagem(loadingMessageId);
    if (data.error) {
      adicionarMensagem("bot", "Erro: " + data.error);
    } else {
      adicionarMensagem("bot", data.resposta);
    }
  } catch (error) {
    removerMensagem(loadingMessageId);
    adicionarMensagem("bot", "Erro ao conectar com o servidor.");
  }
}

function adicionarMensagem(remetente, texto) {
  const container = document.getElementById("chat-container");
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", remetente);
  const headerDiv = document.createElement("div");
  headerDiv.classList.add("message-header");
  const iconSpan = document.createElement("div");
  iconSpan.classList.add("icon-bubble");
  iconSpan.textContent = remetente === "user" ? "🙋" : "🤖";
  const titleSpan = document.createElement("span");
  titleSpan.textContent = remetente === "user" ? "Você" : "Assistente IA";
  headerDiv.appendChild(iconSpan);
  headerDiv.appendChild(titleSpan);
  const p = document.createElement("p");
  p.style.whiteSpace = "pre-line";
  const formatted = texto.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  p.innerHTML = formatted;
  const timestamp = document.createElement("span");
  timestamp.classList.add("timestamp");
  const now = new Date();
  timestamp.textContent = now.getHours() + ":" + String(now.getMinutes()).padStart(2, '0');
  messageDiv.appendChild(headerDiv);
  messageDiv.appendChild(p);
  messageDiv.appendChild(timestamp);
  container.appendChild(messageDiv);
  container.scrollTop = container.scrollHeight;
  const id = "msg-" + Date.now();
  messageDiv.setAttribute("id", id);
  return id;
}

function removerMensagem(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}


// Função para atualizar os dados dos sensores (mantida como no seu original)
function atualizarSensores() {
    let dadosAtuais = {};
    function atualizar() {
        fetch('/api/sensores')
            .then(response => {
                if (!response.ok) throw new Error('Erro ao obter dados dos sensores');
                return response.json();
            })
            .then(data => {
                const mapeamento = {
                    "frequencia-respiratoria": "sensor-1",
                    "saturacao": "sensor-2",
                    "umidade": "sensor-3",
                    "temperatura-corporal": "sensor-4"
                };
                data.forEach(sensor => {
                    const cardId = mapeamento[sensor.id];
                    if (cardId) {
                        const card = document.getElementById(cardId);
                        if (card) {
                            const valorElement = card.querySelector('.sensor-value');
                            if (valorElement && dadosAtuais[sensor.id] !== sensor.valor) {
                                valorElement.innerHTML = `${sensor.valor} <span style="font-size: 0.9rem;">${sensor.unidade}</span>`;
                                dadosAtuais[sensor.id] = sensor.valor;
                            }
                        }
                    }
                });
            })
            .catch(error => console.error('Erro ao atualizar sensores:', error));
    }
    atualizar();
    setInterval(atualizar, 5000);
}


async function atualizarDadosAmbientais() {
  console.log("Buscando dados ambientais...");
  try {
    const response = await fetch("/api/dados-ambientais");
    if (!response.ok) {
      throw new Error(`Erro na rede: ${response.statusText}`);
    }
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    // ✅ CORREÇÃO: A chave para qualidade do ar agora é "AQI", exatamente como o backend envia.
    const cardMapping = {
      "AQI": "qualidade-ar-card",
      "Umidade": "umidade-card",
      "Temperatura": "clima-card"
    };
    
    // Zera o estado visual dos cards para o usuário ver que está atualizando
    Object.values(cardMapping).forEach(cardId => {
      const cardElement = document.getElementById(cardId);
      if (cardElement) {
        const pElement = cardElement.querySelector("p");
        if (pElement) pElement.textContent = "Carregando...";
      }
    });

    // Preenche os cards com os dados recebidos
    data.sugestoes_ambientais.forEach(sugestao => {
      const cardId = cardMapping[sugestao.condicao];
      if (!cardId) return; // Pula dados que não têm um card correspondente (ex: PM2.5 se não houver card para ele)

      const cardElement = document.getElementById(cardId);
      if (!cardElement) return;

      const pElement = cardElement.querySelector("p");
      if (pElement) {
        pElement.innerHTML = `<strong>${sugestao.valor}</strong> - ${sugestao.recomendacao}`;
      }

      // Reseta as classes de cor e aplica a correta
      cardElement.className = 'insight-card'; 
      switch (sugestao.emoji) {
        case "🟢": cardElement.classList.add("positive"); break;
        case "🟡": case "🟠": cardElement.classList.add("warning"); break;
        case "🔴": case "🟣": cardElement.classList.add("alert"); break;
        case "🔵": cardElement.classList.add("info"); break;
      }
    });

  } catch (error) {
    console.error("Erro ao atualizar dados ambientais:", error);
    document.querySelectorAll(".insight-card p").forEach(p => {
      p.textContent = "Erro ao carregar.";
    });
  }
}

// Funções de sugestões de perguntas (sem alterações)
function atualizarQuickQuestions(perguntas) {
  const container = document.getElementById('quick-questions');
  if (!container) return;
  container.innerHTML = ''; 

  perguntas.forEach(texto => {
      const btn = document.createElement('button');
      btn.className = 'pulse-animation';
      btn.textContent = texto;
      btn.onclick = () => enviarPergunta(texto);
      container.appendChild(btn);
  });
}

async function carregarSugestoesIniciais() {
  try {
      const response = await fetch('/api/sugestoes-iniciais');
      const data = await response.json();
      if (data.sugestoes) {
          atualizarQuickQuestions(data.sugestoes);
      }
  } catch (error) {
      console.error('Erro ao carregar sugestões iniciais:', error);
  }
}

// --- INICIALIZAÇÃO DA PÁGINA ---
window.addEventListener('DOMContentLoaded', () => {
  // Carrega as sugestões de perguntas
  carregarSugestoesIniciais();

  // ✅ CORREÇÃO: Chama as funções de atualização de dados imediatamente ao carregar a página
  atualizarDadosAmbientais(); 
  atualizarSensores();

  // Configura as atualizações automáticas (timers)
  setInterval(atualizarDadosAmbientais, 300000); // A cada 5 minutos
});


//designnn
function renderBotMessage(content) {
  const chatContainer = document.getElementById('chatContainer');
  
  const messageElement = document.createElement('div');
  messageElement.classList.add('bot-message');

  messageElement.innerHTML = `
    <div class="chat-bubble">
      <div class="chat-header">
        🤖 <strong>Assistente IA</strong>
      </div>
      <div class="chat-content">
        ${content}
      </div>
      <div class="chat-footer">
        <small>14:31</small>
      </div>
    </div>
  `;

  chatContainer.appendChild(messageElement);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}
