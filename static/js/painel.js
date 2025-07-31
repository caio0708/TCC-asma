document.addEventListener('DOMContentLoaded', function() {
  // Inicialização dos gráficos
  const charts = {};

  function initChart(id, config) {
    const ctx = document.getElementById(id);
    if (ctx) {
      charts[id] = new Chart(ctx, config);
    } else {
      console.error(`Canvas com ID '${id}' não encontrado.`);
    }
  }

    // 1) Gráfico de qualidade do ar
    initChart('airQualityChart', {
      type: 'bar',
      data: {
        labels: airLabels,
        datasets: [{
          label: 'Qualidade do Ar',
          data: airData,
          backgroundColor: ['#EF4444', '#F97316', '#FCD34D']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } }
      }
    });
  
    // 2) Gráfico de sintomas vs umidade
    initChart('umidVsSymptomsChart', {
      data: {
        labels: umidLabels,
        datasets: [
          {
            type: 'line',
            label: 'Umidade (%)',
            data: umidQuality,
            borderColor: '#0EA5E9',
            backgroundColor: 'rgba(14, 165, 233, 0.2)',
            fill: false,
            yAxisID: 'y1',
            tension: 0.3
          },
          {
            type: 'bar',
            label: 'Intensidade dos Sintomas',
            data: umidSymptoms,
            backgroundColor: '#FACC15',
            yAxisID: 'y'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y:  { type: 'linear', position: 'left', beginAtZero: true, title: { display: true, text: 'Sintomas' }},
          y1: { type: 'linear', position: 'right', beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: 'Umidade (%)' }}
        },
        plugins: { legend: { display: true } }
      }
    });
  
    // 3) Gráfico de predição de crises
const crise = crisesPrediction;  
const data = [crise];

initChart('crisisPredictionChart', {
  type: 'bar',
  data: {
    labels: ['Estado'],
    datasets: [{
      label: 'Crise',
      data: data,
      backgroundColor: crise === 1 ? '#F87171' : '#34D399'
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 1,
          callback: v => v === 1 ? 'Crise' : 'Sem Crise'
        }
      }
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: `Usuário: ${username} | Acurácia: ${(accuracy * 100).toFixed(2)}% | Previsão: ${crisesPrediction} | Atualizado em: ${hora} `,
        font: { size: 16 },
        padding: { top: 10, bottom: 20 }
      }
    }
  }
});



  });

// Função para buscar dados dos gráficos
function buscarDadosGraficos() {
  return fetch('/api/graficos')
    .then(response => {
      if (!response.ok) {
        throw new Error('Erro ao obter dados dos gráficos');
      }
      return response.json();
    })
    .then(data => {
      return {
        airQuality: {
          labels: data.air_quality.labels,
          data: data.air_quality.data
        },
        umidVsSymptoms: {
          labels: data.umid_vs_symptoms.labels,
          umidQuality: data.umid_vs_symptoms.umid_quality,
          symptoms: data.umid_vs_symptoms.symptoms
        },
        crisisPrediction: {
          data: data.crisis_probabilities
        }
      };
    })
    .catch(error => {
      console.error('Erro ao buscar dados dos gráficos:', error);
    });
}

// Função para atualizar os gráficos
function atualizarGraficos() {
  buscarDadosGraficos().then(dados => {
    if (dados) {
      // Atualizar gráfico de qualidade do ar
      if (charts['airQualityChart']) {
        charts['airQualityChart'].data.labels = dados.airQuality.labels;
        charts['airQualityChart'].data.datasets[0].data = dados.airQuality.data;
        charts['airQualityChart'].update();
      }

      // Atualizar gráfico de umidade vs sintomas
      if (charts['umidVsSymptomsChart']) {
        charts['umidVsSymptomsChart'].data.labels = dados.umidVsSymptoms.labels;
        charts['umidVsSymptomsChart'].data.datasets[0].data = dados.umidVsSymptoms.umidQuality;
        charts['umidVsSymptomsChart'].data.datasets[1].data = dados.umidVsSymptoms.symptoms;
        charts['umidVsSymptomsChart'].update();
      }

      // Atualizar gráfico de predição de crises
      if (charts['crisisPredictionChart']) {
        charts['crisisPredictionChart'].data.datasets[0].data = dados.crisisPrediction.data;
        charts['crisisPredictionChart'].update();
      }
    }
  });
}

  // Alternar entre gráficos e atualizar os gráficos quando visíveis
  const tabButtons = document.querySelectorAll('.graph-tab-btn');
  const canvases = document.querySelectorAll('.graph-canvas');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', function() {
      tabButtons.forEach(b => b.classList.remove('active'));
      canvases.forEach(c => c.classList.remove('active'));

      this.classList.add('active');

      const targetId = this.getAttribute('data-target');
      const targetCanvas = document.getElementById(targetId);
      if (targetCanvas) {
        targetCanvas.classList.add('active');
        if (charts[targetId]) {
          // Atualiza o gráfico ao mudar de aba
          charts[targetId].resize();
          charts[targetId].update();
        }
      } else {
        console.error(`Canvas com ID '${targetId}' não encontrado.`);
      }
    });

  // Executa a atualização inicial
  atualizarGraficos();

  // Configura atualizações automáticas a cada 5 segundos
  setInterval(() => {
    atualizarGraficos();
  }, 5000);

});

// Função para atualizar os valores dos sensores
// Objeto para armazenar os valores atuais dos sensores e evitar atualizações desnecessárias
let dadosAtuais = {};

// Função para atualizar os valores dos sensores
function atualizarSensores() {
    fetch('/api/sensores') // Requisição à API que fornece os dados dos sensores
        .then(response => {
            if (!response.ok) {
                throw new Error('Erro ao obter dados dos sensores');
            }
            return response.json(); // Converte a resposta para JSON
        })
        .then(data => {
            // Hora atual formatada para o campo "Atualizado"
            const horaAtual = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });

            // Mapeamento dos IDs dos sensores para os IDs dos cartões
            const mapeamento = {
                "frequencia-respiratoria": "card-1",
                "saturacao": "card-2",
                "contagem-tosse": "card-3"
            };

            // Atualiza cada sensor no cartão correspondente
            data.forEach(sensor => {
                const cardId = mapeamento[sensor.id];
                if (cardId) {
                    const card = document.getElementById(cardId);
                    if (card) {
                        // Atualiza o valor e a unidade
                        const valorElement = card.querySelector('.card-value');
                        if (valorElement && dadosAtuais[sensor.id] !== sensor.valor) {
                            valorElement.innerHTML = `${sensor.valor} <span style="font-size: 0.9rem;">${sensor.unidade}</span>`;
                            dadosAtuais[sensor.id] = sensor.valor;
                        }

                        // Atualiza o horário
                        const updatedElement = card.querySelector('.card-updated');
                        if (updatedElement) {
                            updatedElement.textContent = `Atualizado: ${horaAtual}`;
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Erro ao atualizar sensores:', error);
        });
}

// Executa a atualização ao carregar a página
atualizarSensores();

// Configura atualizações automáticas a cada 5 segundos
setInterval(atualizarSensores, 5000);