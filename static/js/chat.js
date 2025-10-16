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
  // ✅ MELHORIA: Processa o texto para suportar negrito e quebras de linha
  // Substitui **texto** por <strong>texto</strong>
  let formattedHtml = texto.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Substitui quebras de linha (\n) por <br> para renderização em HTML
  formattedHtml = formattedHtml.replace(/\n/g, '<br>');
  p.innerHTML = formattedHtml;

  const timestamp = document.createElement("span");
  timestamp.classList.add("timestamp");
  const now = new Date();
  timestamp.textContent = now.getHours() + ":" + String(now.getMinutes()).padStart(2, '0');

  messageDiv.appendChild(headerDiv);
  messageDiv.appendChild(p);
  messageDiv.appendChild(timestamp);

  container.appendChild(messageDiv);
  // Garante que a conversa role para a mensagem mais recente
  container.scrollTop = container.scrollHeight;
  
  const id = "msg-" + Date.now();
  messageDiv.setAttribute("id", id);
  return id;
}

// ✅ CORREÇÃO: Removido o '}' extra que fechava o escopo do arquivo incorretamente.
// A função renderBotMessage não existia e foi removida.

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
                    "batimentos-cardiacos": "sensor-1",
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

    const cardMapping = {
      "AQI": "qualidade-ar-card",
      "Umidade": "umidade-card",
      "Temperatura": "clima-card"
    };
    
    Object.values(cardMapping).forEach(cardId => {
      const cardElement = document.getElementById(cardId);
      if (cardElement) {
        const pElement = cardElement.querySelector("p");
        if (pElement) pElement.textContent = "Carregando...";
      }
    });

    data.sugestoes_ambientais.forEach(sugestao => {
      const cardId = cardMapping[sugestao.condicao];
      if (!cardId) return; 

      const cardElement = document.getElementById(cardId);
      if (!cardElement) return;

      const pElement = cardElement.querySelector("p");
      if (pElement) {
        pElement.innerHTML = `<strong>${sugestao.valor}</strong> - ${sugestao.recomendacao}`;
      }

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

// Funções de sugestões de perguntas 
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
  carregarSugestoesIniciais();
  atualizarDadosAmbientais(); 
  atualizarSensores();
  setInterval(atualizarDadosAmbientais, 300000);
});

// ✅ CORREÇÃO: Removida a função 'renderBotMessage' e a chave '}' extra que estavam aqui.