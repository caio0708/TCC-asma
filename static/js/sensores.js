document.addEventListener("DOMContentLoaded", () => {
  // Atualiza os sensores imediatamente após carregar o DOM
  atualizarSensores();

  // Configura um intervalo para atualizar os sensores
  setInterval(atualizarSensores, 1000);
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

    if (card) {
      const valorElemento = card.querySelector('.card-value');
      const horarioElemento = card.querySelector('.card-updated');

      if (valorElemento) {
        valorElemento.innerHTML = `${sensor.valor} <span style="font-size: 0.9rem;">${sensor.unidade}</span>`;
      }

      if (horarioElemento) {
        horarioElemento.textContent = `Atualizado: ${agora}`;
      }
    } else {
      console.warn(`Card com data-id="${sensor.id}" não encontrado.`);
    }
  });
}