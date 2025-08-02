// painel.js (CORRIGIDO)

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
    
    // A função para atualizar os cards pode ser mantida, mas não resolve o problema do gráfico.
    // O ideal seria unificar as fontes de dados, mas vamos focar no problema do gráfico.
    function atualizarCardsSuperiores() {
        // ... (seu código original aqui, se necessário) ...
        // Nota: Esta função pode se tornar redundante se a API principal for rápida o suficiente.
    }

    // Inicialização dos gráficos com dados do Jinja2 (do carregamento inicial da página)
    initChart('airQualityChart', {
        type: 'bar',
        data: {
            labels: airLabels, // Vindo do painel.html
            datasets: [{
                label: 'Qualidade do Ar',
                data: airData, // Vindo do painel.html
                backgroundColor: ['#EF4444', '#F97316', '#FCD34D']
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } },
            plugins: { legend: { display: false } }
        }
    });

    initChart('historicalDataChart', {
        type: 'line',
        data: {
            labels: historicalLabels, // Vindo do painel.html
            datasets: [
                {
                    label: 'Temperatura Corporal (°C)',
                    data: bodyTempData, // Vindo do painel.html
                    borderColor: '#f97316', yAxisID: 'y_temp', tension: 0.3, spanGaps: true
                },
                {
                    label: 'Temperatura Ambiente (°C)',
                    data: ambientTempData, // Vindo do painel.html
                    borderColor: '#3b82f6', yAxisID: 'y_temp', tension: 0.3, spanGaps: true
                },
                {
                    label: 'Umidade (%)',
                    data: humidityData, // Vindo do painel.html
                    borderColor: '#10b981', yAxisID: 'y_percent', tension: 0.3, spanGaps: true
                },
                {
                    label: 'Saturação Oxi (%)',
                    data: spo2Data, // Vindo do painel.html
                    borderColor: '#ef4444', yAxisID: 'y_percent', tension: 0.3, spanGaps: true
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
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
                data: [crisesPrediction], // Vindo do painel.html
                backgroundColor: crisesPrediction === 1 ? '#F87171' : '#34D399'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1, ticks: { stepSize: 1, callback: v => v === 1 ? 'Crise' : 'Sem Crise' } } },
            plugins: { title: { display: true, text: `Usuário: ${username} | Acurácia: ${(accuracy * 100).toFixed(2)}% | Atualizado: ${hora}` } }
        }
    });

    // [FUNÇÃO CORRIGIDA] para buscar e aplicar os dados da nova API
    function atualizarGraficos() {
        // A rota agora existe e retorna JSON com todos os dados necessários
        fetch('/painel/data') 
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Erro na rede: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                // Atualiza o gráfico de Qualidade do Ar
                const airChart = charts['airQualityChart'];
                if (airChart) {
                    airChart.data.datasets[0].data = data.air_quality.values;
                }

                // Atualiza o gráfico Histórico
                const historicalChart = charts['historicalDataChart'];
                if (historicalChart) {
                    // Atualiza os rótulos do eixo X (horas)
                    historicalChart.data.labels = data.historical_chart_data.labels;
                    // Atualiza cada conjunto de dados
                    historicalChart.data.datasets[0].data = data.historical_chart_data.body_temp;
                    historicalChart.data.datasets[1].data = data.historical_chart_data.ambient_temp;
                    historicalChart.data.datasets[2].data = data.historical_chart_data.humidity;
                    historicalChart.data.datasets[3].data = data.historical_chart_data.spo2;
                }
                
                // Atualiza o gráfico de Predição de Crise
                const crisisChart = charts['crisisPredictionChart'];
                if (crisisChart) {
                    crisisChart.data.datasets[0].data = [data.crisesPrediction];
                    crisisChart.data.datasets[0].backgroundColor = data.crisesPrediction === 1 ? '#F87171' : '#34D399';
                    crisisChart.options.plugins.title.text = `Usuário: ${username} | Acurácia: ${(data.accuracy * 100).toFixed(2)}% | Atualizado: ${data.hora}`;
                }

                // Aplica as atualizações a todos os gráficos de uma vez
                Object.values(charts).forEach(chart => chart.update());
            })
            .catch(error => console.error('Erro ao atualizar dados dos gráficos:', error));
    }

    // --- CHAMADAS DE ATUALIZAÇÃO ---
    // Atualiza os GRÁFICOS a cada 10 segundos
    setInterval(atualizarGraficos, 10000); 
    
    // A outra chamada pode ser mantida se ela atualizar outros elementos na página
    // setInterval(atualizarCardsSuperiores, 1000);

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