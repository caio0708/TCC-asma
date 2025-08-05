// input_file_1.js (CORRIGIDO E COMPLETO)

document.addEventListener('DOMContentLoaded', function() {
    const charts = {};

    function initChart(id, config) {
        const ctx = document.getElementById(id);
        if (ctx) {
            charts[id] = new Chart(ctx, config);
        } else {
            console.error(`Canvas com ID '${id}' não encontrado.`);
        }
    }

    // --- FUNÇÃO ADICIONADA PARA ATUALIZAR OS CARDS SUPERIORES ---
    function atualizarCardsSuperiores() {
        fetch('/api/sensores') // Busca da mesma API que a página de sensores
            .then(response => {
                if (!response.ok) {
                    throw new Error('Erro ao buscar dados dos sensores para os cards');
                }
                return response.json();
            })
            .then(sensores => {
                const agora = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });

                // Mapeia IDs dos sensores para os IDs dos elementos HTML
                const sensorMap = {
                    'temperatura-corporal': { valId: 'valor-temperatura-corporal', statId: 'status-temperatura-corporal', precision: 1 },
                    'temperatura-ambiente': { valId: 'valor-temperatura-ambiente', statId: 'status-temperatura-ambiente', precision: 1 },
                    'umidade': { valId: 'valor-umidade', statId: 'status-umidade', precision: 0 },
                    'saturacao': { valId: 'valor-saturacao', statId: 'status-saturacao', precision: 0 },
                    'frequencia-respiratoria': { valId: 'valor-freq-resp', statId: 'status-freq-resp', precision: 0 },
                    'contagem-tosse': { valId: 'valor-contagem-tosse', statId: 'status-contagem-tosse', precision: 0 }
                };

                sensores.forEach(sensor => {
                    if (sensorMap[sensor.id]) {
                        const { valId, statId, precision } = sensorMap[sensor.id];
                        const valorElemento = document.getElementById(valId);
                        const statusElemento = document.getElementById(statId);

                        if (valorElemento) {
                            valorElemento.textContent = parseFloat(sensor.valor).toFixed(precision);
                        }
                        if (statusElemento) {
                            statusElemento.textContent = `Atualizado: ${agora}`;
                        }
                    }
                });
            })
            .catch(error => console.error('Erro ao atualizar cards superiores:', error));
    }


    // Inicializar gráficos com dados iniciais (código original mantido)
    initChart('airQualityChart', {
        type: 'bar',
        data: {
            labels: airLabels,
            datasets: [{
                label: 'Qualidade do Ar',
                data: airData.map(v => (v === null || isNaN(v)) ? 0 : v),
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

    initChart('historicalDataChart', {
        type: 'line',
        data: {
            labels: historicalLabels,
            datasets: [
                {
                    label: 'Temperatura Corporal (°C)',
                    data: bodyTempData.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#f97316',
                    yAxisID: 'y_temp',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Temperatura Ambiente (°C)',
                    data: ambientTempData.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#3b82f6',
                    yAxisID: 'y_temp',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Umidade (%)',
                    data: humidityData.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#10b981',
                    yAxisID: 'y_percent',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Saturação Oxi (%)',
                    data: spo2Data.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#ef4444',
                    yAxisID: 'y_percent',
                    tension: 0.3,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y_temp: { type: 'linear', position: 'left', title: { display: true, text: 'Temperatura (°C)' } },
                y_percent: { type: 'linear', position: 'right', title: { display: true, text: 'Porcentagem (%)' }, min: 0, max: 100, grid: { drawOnChartArea: false } }
            }
        }
    });

    initChart('crisisPredictionChart', {
        type: 'bar',
        data: {
            labels: ['Estado Atual'],
            datasets: [{
                label: 'Predição de Crise',
                data: [crisesPrediction === null || isNaN(crisesPrediction) ? 0 : crisesPrediction],
                backgroundColor: crisesPrediction === 1 ? '#F87171' : '#34D399'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1, ticks: { stepSize: 1, callback: v => v === 1 ? 'Crise' : 'Sem Crise' } } },
            plugins: { title: { display: true, text: `Usuário: ${username} | Acurácia: ${(accuracy * 100).toFixed(2)}% | Atualizado: ${hora}` } }
        }
    });

    // Função original para atualizar os gráficos
    function atualizarSensores() {
        fetch('/painel/data')
            .then(response => response.json())
            .then(data => {
                charts['airQualityChart'].data.datasets[0].data = data.airData.map(v => (v === null || isNaN(v)) ? 0 : v);
                charts['historicalDataChart'].data.datasets[0].data = data.bodyTempData.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[1].data = data.ambientTempData.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[2].data = data.humidityData.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[3].data = data.spo2Data.map(v => (v === null || isNaN(v)) ? null : v);
                charts['crisisPredictionChart'].data.datasets[0].data = [data.crisesPrediction === null || isNaN(data.crisesPrediction) ? 0 : data.crisesPrediction];
                charts['crisisPredictionChart'].data.datasets[0].backgroundColor = data.crisesPrediction === 1 ? '#F87171' : '#34D399';
                charts['crisisPredictionChart'].options.plugins.title.text = `Usuário: ${username} | Acurácia: ${(data.accuracy * 100).toFixed(2)}% | Atualizado: ${new Date().toLocaleTimeString('pt-BR')}`;
                Object.values(charts).forEach(chart => chart.update());
            })
            .catch(error => console.error('Erro ao obter dados dos sensores:', error));
    }

    // --- CHAMADAS DE ATUALIZAÇÃO ---
    // Atualiza os GRÁFICOS imediatamente e a cada 5 segundos
    atualizarSensores();
    setInterval(atualizarSensores, 5000);
    
    // Atualiza os CARDS SUPERIORES imediatamente e a cada 1 segundos
    atualizarCardsSuperiores();
    setInterval(atualizarCardsSuperiores, 1000);

    // Lógica das abas (sem alteração)
    const tabButtons = document.querySelectorAll('.graph-tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.graph-canvas, .graph-tab-btn').forEach(el => el.classList.remove('active'));
            this.classList.add('active');
            const targetCanvas = document.getElementById(this.getAttribute('data-target'));
            if (targetCanvas) {
                targetCanvas.classList.add('active');
                if (charts[targetCanvas.id]) {
                    charts[targetCanvas.id].resize();
                }
            }
        });
    });
});