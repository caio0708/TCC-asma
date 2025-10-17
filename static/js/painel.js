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
    // 1. Defina os níveis de risco para cada poluente (exemplo, valores em µg/m³)
    const airQualityThresholds = {
        'AQI':      { moderate: 2, bad: 4 },
        'PM2.5':    { moderate: 10, bad: 25 },
        'PM10':     { moderate: 20, bad: 50 },
        'O3':       { moderate: 60, bad: 120 },
        'NO2':      { moderate: 40, bad: 100 },
        'SO2':      { moderate: 20, bad: 50 }
    };

    // 2. Função para obter a cor com base no valor e no poluente
    function getBarColor(pollutant, value) {
        const thresholds = airQualityThresholds[pollutant];
        if (!thresholds) return '#A5B4FC'; // Cor padrão

        if (value >= thresholds.bad) {
            return '#EF4444'; // Vermelho (Ruim)
        }
        if (value >= thresholds.moderate) {
            return '#FBBF24'; // Amarelo (Moderado)
        }
        return '#34D399'; // Verde (Bom)
    }

    // 3. Calcule as cores dinamicamente antes de criar o gráfico
    const backgroundColors = airLabels.map((label, index) => {
        return getBarColor(label, airData[index]);
    });

    // 4. Crie o gráfico
    initChart('airQualityChart', {
        type: 'bar',
        data: {
            labels: airLabels,
            datasets: [{
                label: 'Qualidade do Ar',
                data: airData.map(v => (v === null || isNaN(v)) ? 0 : v),
                // --- COR DINÂMICA APLICADA AQUI ---
                backgroundColor: backgroundColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                // Você pode combinar esta opção com a escala logarítmica da Opção 1!
                y: {
                    type: 'logarithmic', // Combinação poderosa!
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
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
                    borderColor: '#ec7017ff',
                    yAxisID: 'y_temp',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Temperatura Ambiente (°C)',
                    data: ambientTempData.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#f3d421ff',
                    yAxisID: 'y_temp',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Umidade (%)',
                    data: humidityData.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#208ff7ff',
                    yAxisID: 'y_percent',
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'Saturação Oxi (%)',
                    data: spo2Data.map(v => (v === null || isNaN(v)) ? null : v),
                    borderColor: '#eb2121ff',
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
    
    // [CORRIGIDO] Lógica para formatar a acurácia de forma segura
    const formatAccuracy = (acc) => {
        if (typeof acc === 'number' && !isNaN(acc)) {
            return `${(acc * 100).toFixed(1)}%`;
        }
        return 'N/A'; // Retorna N/A se o valor for inválido
    };
    
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
            plugins: { title: { display: true, text: `Usuário: ${username} | Acurácia: ${formatAccuracy(accuracy)} | Atualizado: ${hora}` } }
        }
    });

    // Função original para atualizar os gráficos
    function atualizarSensores() {
        fetch('/painel/data')
            .then(response => response.json())
            .then(data => {
                charts['airQualityChart'].data.datasets[0].data = data.air_quality.values.map(v => (v === null || isNaN(v)) ? 0 : v);
                charts['historicalDataChart'].data.datasets[0].data = data.historical_chart_data.body_temp.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[1].data = data.historical_chart_data.ambient_temp.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[2].data = data.historical_chart_data.humidity.map(v => (v === null || isNaN(v)) ? null : v);
                charts['historicalDataChart'].data.datasets[3].data = data.historical_chart_data.spo2.map(v => (v === null || isNaN(v)) ? null : v);
                charts['crisisPredictionChart'].data.datasets[0].data = [data.crisesPrediction === null || isNaN(data.crisesPrediction) ? 0 : data.crisesPrediction];
                charts['crisisPredictionChart'].data.datasets[0].backgroundColor = data.crisesPrediction === 1 ? '#F87171' : '#34D399';

                // [CORRIGIDO] Usa a função de formatação segura aqui também
                const accuracyText = formatAccuracy(data.accuracy);
                const updateTime = data.hora || new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
                charts['crisisPredictionChart'].options.plugins.title.text = `Usuário: ${username} | Acurácia: ${accuracyText} | Atualizado: ${updateTime}`;
                
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