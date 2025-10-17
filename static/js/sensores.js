document.addEventListener("DOMContentLoaded", () => {
  // Atualiza os sensores imediatamente após carregar o DOM
  atualizarSensores();

  // Configura um intervalo para atualizar os sensores
  setInterval(atualizarSensores, 500);
});

function atualizarSensores() {
  fetch('/api/sensores')
    .then(response => {
      if (!response.ok) {
        throw new Error('Erro ao buscar dados dos sensores');
      }
      return response.json();
    })
    .then(sensores => atualizarValoresDosSensores(sensores))
    .catch(error => console.error('Erro ao obter dados dos sensores:', error));
}

function atualizarValoresDosSensores(sensores) {
  const agora = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Tratamento especial para o Acelerômetro
  const acelerometroCard = document.querySelector('.card[data-id="acelerometro-x"]');
  if (acelerometroCard) {
    const valorX = sensores.find(s => s.id === 'acelerometro-x');
    const valorY = sensores.find(s => s.id === 'acelerometro-y');
    const valorZ = sensores.find(s => s.id === 'acelerometro-z');

    if (valorX) {
      const xElement = acelerometroCard.querySelector('div[data-id="acelerometro-x"] span:first-child');
      xElement.nextSibling.textContent = `${valorX.valor} `;
    }
    if (valorY) {
      const yElement = acelerometroCard.querySelector('div[data-id="acelerometro-y"] span:first-child');
      yElement.nextSibling.textContent = `${valorY.valor} `;
    }
    if (valorZ) {
      const zElement = acelerometroCard.querySelector('div[data-id="acelerometro-z"] span:first-child');
      zElement.nextSibling.textContent = `${valorZ.valor} `;
    }
    
    const horarioElemento = acelerometroCard.querySelector('.card-updated');
    if (horarioElemento) {
      horarioElemento.textContent = `Atualizado: ${agora}`;
    }
  }

  // Tratamento especial para o Giroscópio
  const giroscopioCard = document.querySelector('.card[data-id="giroscopio-x"]');
  if (giroscopioCard) {
    const valorRoll = sensores.find(s => s.id === 'giroscopio-x'); // Roll
    const valorPitch = sensores.find(s => s.id === 'giroscopio-y'); // Pitch
    const valorYaw = sensores.find(s => s.id === 'giroscopio-z'); // Yaw

    if (valorRoll) {
        const rollElement = giroscopioCard.querySelector('div[data-id="giroscopio-x"] span:first-child');
        rollElement.nextSibling.textContent = `${valorRoll.valor} `;
    }
    if (valorPitch) {
        const pitchElement = giroscopioCard.querySelector('div[data-id="giroscopio-y"] span:first-child');
        pitchElement.nextSibling.textContent = `${valorPitch.valor} `;
    }
    if (valorYaw) {
        const yawElement = giroscopioCard.querySelector('div[data-id="giroscopio-z"] span:first-child');
        yawElement.nextSibling.textContent = `${valorYaw.valor} `;
    }

    const horarioElemento = giroscopioCard.querySelector('.card-updated');
    if (horarioElemento) {
      horarioElemento.textContent = `Atualizado: ${agora}`;
    }
  }

  // Lógica para os outros sensores (individuais)
  sensores.forEach(sensor => {
    // Evita reprocessar os eixos do acelerômetro e giroscópio individualmente
    if (sensor.id.startsWith('acelerometro-') || sensor.id.startsWith('giroscopio-')) {
      return;
    }

    const card = document.querySelector(`.card[data-id="${sensor.id}"]`);

// Dentro da função atualizarValoresDosSensores()
if (card) {
  const valorElemento = card.querySelector('.card-value');
  const horarioElemento = card.querySelector('.card-updated');

  if (valorElemento) {
    valorElemento.innerHTML = `${sensor.valor} <span style="font-size: 0.9rem;">${sensor.unidade}</span>`;
  }

  if (horarioElemento) {
    horarioElemento.textContent = `Atualizado: ${agora}`;
  }

  // --- NOVO: aplica a classe de status ---
  const statusClass = getSensorStatus(sensor.id, sensor.valor);
  card.classList.remove('is-ok', 'is-warn', 'is-bad'); // limpa anteriores
  if (statusClass) {
    card.classList.add(statusClass);
  }
}
  });
}

function getSensorStatus(id, valor) {
    const v = parseFloat(valor);
    if (isNaN(v)) return ''; // Não aplica status se o valor não for numérico

    switch (id) {
        case 'batimentos-cardiacos':
            if (v >= 60 && v <= 100) return 'is-ok';
            if ((v >= 50 && v <= 59) || (v >= 101 && v <= 120)) return 'is-warn';
            if (v < 50 || v > 120) return 'is-bad';
            break;
        case 'saturacao':
            if (v >= 95 && v <= 100) return 'is-ok';
            if (v >= 91 && v <= 94) return 'is-warn';
            if (v <= 90 ) return 'is-bad';
            break;
        case 'temperatura-corporal':
            if (v >= 36.5 && v <= 37.5) return 'is-ok';
            if (v >= 37.6 && v <= 38) return 'is-warn';
            if (v <= 35.5 || v >= 38.1) return 'is-bad';
            break;
        case 'contagem-tosse':
            if (v >= 0 && v <= 5) return 'is-ok';
            if ((v >= 6 && v <= 14)) return 'is-warn';
            if (v >= 15) return 'is-bad';
            break;
        case 'qualidade-ar-aqi':
            if (v >= 0 && v <= 1) return 'is-ok';
            if (v > 1 && v <= 3) return 'is-warn';
            if (v >= 4) return 'is-bad';
            break;
        case 'qualidade-ar-pm25':
            if (v >= 0 && v <= 12) return 'is-ok';
            if (v >= 12.1 && v <= 35.4) return 'is-warn';
            if (v >= 35.5) return 'is-bad';
            break;
        case 'qualidade-ar-pm10':
             if (v >= 0 && v <= 54) return 'is-ok';
             if ((v >= 55 && v <= 154)) return 'is-warn';
             if (v >= 155 ) return 'is-bad';
             break;
        case 'qualidade-ar-o3':
            if (v >= 0 && v <= 54) return 'is-ok';
            if (v >= 55 && v <= 70) return 'is-warn';
            if (v >= 71) return 'is-bad';
            break;
        case 'qualidade-ar-no2':
            if (v >= 0 && v <= 53) return 'is-ok';
            if (v >= 54 && v <= 100) return 'is-warn';
            if (v >= 101) return 'is-bad';
            break;
        case 'qualidade-ar-so2':
            if (v >= 0 && v <= 35) return 'is-ok';
            if (v >= 36 && v <= 75) return 'is-warn';
            if (v >= 76 ) return 'is-bad';
            break;
        case 'temperatura-ambiente':
            if (v >= 18 && v <= 24) return 'is-ok';
            if ((v >= 12 && v <= 17) || (v >= 25 && v <= 29)) return 'is-warn';
            if ((v <= 12) || (v >= 30 )) return 'is-bad';
            break;
        case 'umidade':
            if (v >= 40 && v <= 60) return 'is-ok';
            if ((v >= 30 && v <= 39) || (v >= 61 && v <= 70)) return 'is-warn';
            if ((v <= 30) || (v >= 70 )) return 'is-bad';
            break;
        default:
            return ''; // Nenhum status para sensores não listados
    }
    return '';
}