document.addEventListener('DOMContentLoaded', () => {
  // Inicializa um objeto para manter as instâncias dos gráficos
  let calendar;
  // Objeto para armazenar instâncias de gráficos e seus loops de atualização
  const charts = {
    ppgChart: null,
    respirationChart: null,
    soundChart: null,
    motionChart: null,
    piezoChart: null,
    weeklyCoughChart: null
  };
  const chartIntervals = {
    ppgChart: null, soundChart: null, motionChart: null, piezoChart: null
  };

  let isLoadingData = false; // Variavel de controle para evitar race conditions

  // Controle do modo "live dedicado" do Piezo
let piezoUpdateInterval = null;
let isPiezoLiveLoopOn = false;
const PIEZO_MAX_DATA_POINTS = 100;

function initPiezoChart() {
  const canvas = document.getElementById('piezoChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const config = {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Piezo (Mov. Torácico)', data: [], borderWidth: 1.5, pointRadius: 0, tension: 0.1 }
    ]},
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'time', time: { unit: 'second', displayFormats: { second: 'HH:mm:ss' } }, ticks: { maxRotation: 0, autoSkip: true } },
        y: { title: { display: true, text: 'Amplitude do Sinal' }, min: 0, max: 5000 }
      },
      animation: false
    }
  };
  createOrUpdateChart('piezoChart', config);
}

async function fetchPiezoLatest() {
  const chart = charts['piezoChart'];
  if (!chart) return;
  try {
    const res = await fetch('/api/piezo_latest');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const t = new Date(data.timestamp);
    chart.data.labels.push(t);
    chart.data.datasets[0].data.push(data.piezo);
    while (chart.data.labels.length > PIEZO_MAX_DATA_POINTS) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update('none');
  } catch (e) {
    console.error('Falha ao buscar dados do Piezo:', e);
    stopPiezoUpdates();
  }
}

function startPiezoUpdates() {
  stopPiezoUpdates();
  initPiezoChart();
  fetchPiezoLatest();
  piezoUpdateInterval = setInterval(fetchPiezoLatest, 1000); // 1 Hz (igual MPU)
  isPiezoLiveLoopOn = true;
}

function stopPiezoUpdates() {
  if (piezoUpdateInterval) clearInterval(piezoUpdateInterval);
  piezoUpdateInterval = null;
  isPiezoLiveLoopOn = false;
}

  
  // --- MINIGAME LOGIC (sem alterações) ---
  const breathingText = document.getElementById("breathingText"),
        breathingCircle = document.getElementById("breathingCircle"),
        circleProgress = document.querySelector(".circle-progress"),
        startButton = document.getElementById("startGameBtn"),
        downloadContainer = document.getElementById("downloadContainer"),
        predictionResultEl = document.getElementById("predictionResult"),
        radius = 65,
        circumference = 2 * Math.PI * radius;
  if(circleProgress) {
    circleProgress.style.strokeDasharray = `${circumference}`;
    circleProgress.style.strokeDashoffset = `${circumference}`;
  }
  const phases = [
    { name: "inhale", duration: 3000, scale: 1.3, text: "Inspire pela Boca" },
    { name: "hold", duration: 1000, scale: 1.3, text: "Prenda" },
    { name: "exhale", duration: 4000, scale: 0.7, text: "Expire pela Boca" },
    { name: "hold", duration: 1000, scale: 0.7, text: "Prenda" }
  ];
  let phaseIndex = 0, isRunning = false, phaseTimeout = null, mediaRecorder, audioChunks = [];
  
  async function iniciarGravacao() {
    try {
      let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunks = [];
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      
      mediaRecorder.onstop = () => {
        let blob = new Blob(audioChunks, { type: "audio/webm" });
        if (predictionResultEl) {
            predictionResultEl.innerHTML = '<p class="loading">Analisando o áudio...</p>';
            predictionResultEl.style.display = 'block';
        }
        let formData = new FormData();
        formData.append("audio", blob, "gravacao_respiracao.webm");

        fetch("/api/upload-audio", { method: "POST", body: formData })
          .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
          .then(({ ok, status, data }) => {
              if (!ok) throw new Error(data.error || `Falha no upload (${status})`);
              return fetch("/api/predict-audio", {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ wav_path: data.wav_path })
              });
          })
          .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
          .then(({ ok, status, data }) => {
              if (!ok) throw new Error(data.error || `Falha na predição (${status})`);
              if (predictionResultEl) {
                  const confidencePercent = (data.confidence * 100).toFixed(1);
                  predictionResultEl.innerHTML = `
                      <p class="result-title">Resultado da Análise:</p>
                      <p class="result-class">${data.predicted_class}</p>
                      <p class="result-confidence">Confiança: ${confidencePercent}%</p>
                  `;
              }
          })
          .catch(err => {
              console.error("❌ Erro no processo de análise de áudio:", err);
              if (predictionResultEl) {
                  predictionResultEl.innerHTML = `<p class="error">Falha ao processar o áudio: ${err.message}</p>`;
              }
          });
      };
      mediaRecorder.start();
    } catch (e) {
      console.error("Erro ao obter mídia:", e);
      alert("Não foi possível acessar o microfone. Verifique as permissões do navegador.");
    }
  }

  function pararGravacao() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  }

  function animatePhase(e) {
    if (!breathingText || !breathingCircle || !circleProgress) return;
    breathingText.textContent = e.text;
    breathingCircle.style.transform = `scale(${e.scale})`;
    circleProgress.style.transition = "none";
    circleProgress.style.strokeDashoffset = `${circumference}`;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        circleProgress.style.transition = `stroke-dashoffset ${e.duration}ms linear`;
        circleProgress.style.strokeDashoffset = "0";
      });
    });
    phaseTimeout = setTimeout(() => {
      phaseIndex = (phaseIndex + 1) % phases.length;
      if (isRunning) animatePhase(phases[phaseIndex]);
    }, e.duration);
  }

  if(startButton){
    startButton.addEventListener("click", async () => {
      if (isRunning) {
        isRunning = false;
        clearTimeout(phaseTimeout);
        breathingCircle.style.transform = "scale(1)";
        circleProgress.style.transition = "none";
        circleProgress.style.strokeDashoffset = `${circumference}`;
        breathingText.textContent = "Pronto?";
        startButton.textContent = "Iniciar Exercício";
        pararGravacao();
      } else {
        if (predictionResultEl) predictionResultEl.style.display = 'none';
        if (downloadContainer) downloadContainer.innerHTML = "";
        
        await iniciarGravacao();
        isRunning = true;
        phaseIndex = 0;
        animatePhase(phases[phaseIndex]);
        startButton.textContent = "Parar Exercício";
      }
    });
  }
  
  /**
   * Updates the main result cards with sensor data.
   */
  function updateResults(results) {
    const container = document.getElementById('analysisResults');
    if(!container) return;
    container.innerHTML = '';
    const metrics = [
      { label: 'Batimentos Cardíacos', value: results['batimentos-cardiacos'] > 0 ? results['batimentos-cardiacos'].toFixed(0) : 'N/A', unit: 'BPM' },
      { label: 'Saturação', value: results['saturacao'] > 0 ? results['saturacao'].toFixed(1) : 'N/A', unit: '%' },
      { label: 'Temperatura Corporal', value: results['temperatura-corporal'] > 0 ? results['temperatura-corporal'].toFixed(2) : 'N/A', unit: '°C' },
      { label: 'Nível de Ruído', value: results['som'] > 0 ? results['som'].toFixed(0) : 'N/A', unit: 'ADC' },
    ];
    metrics.forEach(metric => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.innerHTML = `<p class="value">${metric.value} <span style="font-size: 1.2rem;">${metric.unit}</span></p><p class="label">${metric.label}</p>`;
      container.appendChild(card);
    });
  }

  /**
   * Atualiza o medidor de PERF (Pico de Fluxo Expiratório)
   */
  function updatePerfGauge(pefrData) {
      const perfSection = document.getElementById('perfAnalysis');
      if (!perfSection) return;

      if (pefrData && !pefrData.error) {
          document.getElementById('perfPredicted').textContent = pefrData.predicted_pefr;
          document.getElementById('perfReference').textContent = pefrData.reference_pefr;
          document.getElementById('perfPercentage').textContent = pefrData.percentage.toFixed(1) + '%';
          
          const zoneTextElement = document.getElementById('perfZoneText');
          zoneTextElement.textContent = pefrData.zone;

          const indicator = document.getElementById('perfIndicator');
          // Limita o valor em 100% para não ultrapassar a barra visualmente
          const positionPercentage = Math.min(pefrData.percentage, 100);
          indicator.style.left = positionPercentage + '%';

          // Adiciona a classe de cor correspondente à zona
          const zoneClass = pefrData.zone.toLowerCase();
          zoneTextElement.className = 'zone-label ' + zoneClass;
      } else {
           // Caso não haja dados de PERF, exibe uma mensagem
           const predictedEl = document.getElementById('perfPredicted');
           if (predictedEl) {
              predictedEl.textContent = '--';
              document.getElementById('perfReference').textContent = '--';
              document.getElementById('perfPercentage').textContent = '--%';
              document.getElementById('perfZoneText').textContent = 'Indisponível';
              document.getElementById('perfZoneText').className = 'zone-label';
              document.getElementById('perfIndicator').style.left = '0%';
           }
           console.warn("Dados de PERF indisponíveis ou com erro:", pefrData?.error);
      }
  }


  /**
   * Generic function to create or update a chart efficiently.
   */
  function createOrUpdateChart(chartId, config) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Use Chart.getChart to find any existing chart on the canvas
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
      existingChart.destroy();
    }

    // Create the new chart
    charts[chartId] = new Chart(ctx, config);
  }

  /**
   * Helper function to display a "No Data" message on a canvas.
   */
  function showNoDataMessage(chartId, message = 'Dados insuficientes para exibir o gráfico') {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    
    // Use Chart.getChart to find and destroy any existing chart
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#999';
    ctx.font = '16px "Segoe UI", "Roboto", "Helvetica Neue", sans-serif';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    ctx.restore();
  }

    // --- Helpers comuns para histórico (padroniza e sanitiza pontos) ---
  function toFiniteOrNull(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  function sanitizePoints(points) {
    if (!Array.isArray(points)) return [];
    return points.map(p => {
      // aceita {x, y} ou {t, ax/ay/az/pz/value}
      const x = p.x ?? p.t ?? p.ts ?? null;
      const rawY = p.y ?? p.value ?? p.pz ?? p.ax ?? p.ay ?? p.az ?? null;
      return { x, y: toFiniteOrNull(rawY) };
    });
  }


  /**
   * Converts a timestamp to a relative time string (e.g., "X seconds ago").
   */
  function formatRelativeTime(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diffSeconds = (now - time) / 1000;
    if (diffSeconds < 60) return `${Math.round(diffSeconds)}s atrás`;
    return time.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  // --- CONFIGURAÇÕES GLOBAIS DOS GRÁFICOS DE ANÁLISE ---
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { 
      legend: { position: 'bottom' },
      tooltip: {
        callbacks: {
          title: (tooltipItems) => {
            return formatRelativeTime(tooltipItems[0].label);
          }
        }
      }
    }
  };

  const timeSeriesOptions = {
    ...commonOptions,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'second',
          displayFormats: { second: 'HH:mm:ss' },
          tooltipFormat: 'HH:mm:ss'
        },
        title: { display: true, text: 'Tempo' },
        ticks: {
          source: 'data',
          maxRotation: 0,
          autoSkip: true,
          callback: function(value, index, values) {
            return formatRelativeTime(value);
          }
        }
      },
      y: { title: { display: true } }
    }
  };

  const yAxisLimits = {
    ppgChart: { min: -150, max: 200, title: 'Amplitude Filtrada' },
    respirationChart: { min: -100, max: 300, title: 'Amplitude Modulada' },
    soundChart: { min: 0, max: 4000, title: 'Amplitude do ADC' },
    motionChart: { min: 0, max: 60, title: 'Magnitude (Acel. e Rotação)' },
    piezoChart: { min: 0, max: 2000, title: 'Amplitude do Sinal' }
  };
  // --- FIM DAS CONFIGURAÇÕES GLOBAIS ---


  // --- LÓGICA DE ATUALIZAÇÃO "LIVE" DEDICADA (NOVO) ---
  const LIVE_UPDATE_INTERVAL_MS = 1000; // 1 segundo
  const LIVE_MAX_DATA_POINTS = 100;

  async function fetchAndUpdateLiveChart(chartId, sensorName) {
    const chart = charts[chartId];
    if (!chart) return;

    try {
      const res = await fetch(`/api/latest_analysis_point?sensor=${sensorName}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.x && data.y !== null) {
        const timestamp = new Date(data.x);
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(data.y);

        // Mantém o número de pontos no gráfico
        while (chart.data.labels.length > LIVE_MAX_DATA_POINTS) {
          chart.data.labels.shift();
          chart.data.datasets[0].data.shift();
        }
        chart.update('none');
      }
    } catch (e) {
      console.error(`Falha ao buscar dados live para ${chartId}:`, e);
      stopLiveChartUpdate(chartId); // Para o loop em caso de erro
    }
  }

  function startLiveChartUpdate(chartId, sensorName) {
    stopLiveChartUpdate(chartId); // Garante que não haja loops duplicados
    // Busca e atualiza imediatamente, depois inicia o intervalo
    fetchAndUpdateLiveChart(chartId, sensorName);
    chartIntervals[chartId] = setInterval(() => fetchAndUpdateLiveChart(chartId, sensorName), LIVE_UPDATE_INTERVAL_MS);
  }

  function stopLiveChartUpdate(chartId) {
    if (chartIntervals[chartId]) {
      clearInterval(chartIntervals[chartId]);
      chartIntervals[chartId] = null;
    }
  }
  /**
   * Renders all analysis charts with improved time axis and fixed Y-axis limits.
   * Esta função foi dividida em funções menores (renderPPGChart, etc.)
   */

  // 1. PPG Chart
  function renderPPGChart(chartData) {
    if (chartData && chartData.ppg?.signal?.length > 1) {
      const ppgConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal PPG Filtrado', data: chartData.ppg.signal, borderColor: '#FF6F61', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            { type: 'scatter', label: 'Batimentos Detectados', data: chartData.ppg.peaks, backgroundColor: '#D32F2F', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: { 
          ...timeSeriesOptions, 
          scales: { 
            ...timeSeriesOptions.scales, 
            y: { 
              ...timeSeriesOptions.scales.y, 
              title: { ...timeSeriesOptions.scales.y.title, text: yAxisLimits.ppgChart.title },
              min: yAxisLimits.ppgChart.min,
              max: yAxisLimits.ppgChart.max
            } 
          } 
        }
      };
      createOrUpdateChart('ppgChart', ppgConfig);
    } else {
      showNoDataMessage('ppgChart');
    }
  }

  // 2. Respiration Chart
  function renderRespirationChart(chartData) {
    if (chartData && chartData.respiration?.signal?.length > 1) {
      const respConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal Respiratório Derivado', data: chartData.respiration.signal, borderColor: '#2196F3', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            { type: 'scatter', label: 'Picos Respiratórios', data: chartData.respiration.peaks, backgroundColor: '#1976D2', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: {
          ...commonOptions,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Tempo (s)' } },
            y: { 
              title: { display: true, text: yAxisLimits.respirationChart.title },
              min: yAxisLimits.respirationChart.min,
              max: yAxisLimits.respirationChart.max
            }
          }
        }
      };
      createOrUpdateChart('respirationChart', respConfig);
    } else {
      showNoDataMessage('respirationChart');
    }
  }
  
  // 3. Sound Chart
  function renderSoundChart(chartData) {
    if (chartData && chartData.sound?.signal?.length > 1) {
      const soundConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal do Microfone', data: chartData.sound.signal, borderColor: '#4CAF50', borderWidth: 1.5, pointRadius: 0 },
            { label: 'Limiar de Som', data: chartData.sound.signal.map(p => ({ x: p.x, y: chartData.sound.threshold })), borderColor: '#F44336', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: 'Picos de Som', data: chartData.sound.peaks, backgroundColor: '#2E7D32', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: { 
          ...timeSeriesOptions, 
          scales: { 
            ...timeSeriesOptions.scales, 
            y: { 
              ...timeSeriesOptions.scales.y, 
              title: { ...timeSeriesOptions.scales.y.title, text: yAxisLimits.soundChart.title },
              min: yAxisLimits.soundChart.min,
              max: yAxisLimits.soundChart.max
            } 
          } 
        }
      };
      createOrUpdateChart('soundChart', soundConfig);
    } else {
      showNoDataMessage('soundChart');
    }
  }

  // 4. Motion Chart
  function renderMotionChart(chartData) {
    if (chartData && chartData.motion?.signal?.length > 1) {
      const motionConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Intensidade do Movimento (Acel.)', data: chartData.motion.signal, borderColor: '#6a5acd', backgroundColor: 'rgba(106, 90, 205, 0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 },
            { label: 'Intensidade da Rotação (Giro.)', data: chartData.motion.rotation_signal, borderColor: '#00a896', fill: false, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Limiar de Detecção de Tosse', data: chartData.motion.signal.map(p => ({ x: p.x, y: chartData.motion.threshold })), borderColor: '#aaaaaa', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: 'Picos de Tosse / Mov. Brusco', data: chartData.motion.peaks, backgroundColor: '#dc3545', pointStyle: 'crossRot', radius: 7, borderWidth: 2 }
          ]
        },
        options: { 
          ...timeSeriesOptions, 
          scales: { 
            ...timeSeriesOptions.scales, 
            y: { 
              ...timeSeriesOptions.scales.y, 
              title: { ...timeSeriesOptions.scales.y.title, text: yAxisLimits.motionChart.title },
              min: yAxisLimits.motionChart.min,
              max: yAxisLimits.motionChart.max
            } 
          } 
        }
      };
      createOrUpdateChart('motionChart', motionConfig);
    } else {
      showNoDataMessage('motionChart');
    }
  }

  // 5. Piezo Chart
  function renderPiezoChart(chartData) {
    if (chartData && chartData.piezo?.signal?.length > 1) {
      const piezoConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal Piezo (Mov. Torácico)', data: chartData.piezo.signal, borderColor: '#FF9800', borderWidth: 1.5, pointRadius: 0 },
            { label: 'Limiar de Vibração', data: chartData.piezo.signal.map(p => ({ x: p.x, y: chartData.piezo.threshold })), borderColor: '#F44336', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: 'Picos de Vibração', data: chartData.piezo.peaks, backgroundColor: '#EF6C00', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: { 
          ...timeSeriesOptions, 
          scales: { 
            ...timeSeriesOptions.scales, 
            y: { 
              ...timeSeriesOptions.scales.y, 
              title: { ...timeSeriesOptions.scales.y.title, text: yAxisLimits.piezoChart.title },
              min: yAxisLimits.piezoChart.min,
              max: yAxisLimits.piezoChart.max
            } 
          } 
        }
      };
      createOrUpdateChart('piezoChart', piezoConfig);
    } else {
      showNoDataMessage('piezoChart');
    }
  }
  
  /**
   * Atualiza seletivamente os gráficos de análise "live"
   * Esta função agora é chamada apenas uma vez no carregamento inicial e pelo loop principal
   * para atualizar o gráfico de respiração, que não tem um modo "live" dedicado.
   */
  function updateLiveCharts(analysisData) {
    if (!container || !analysisData || !analysisData.charts) {
      renderRespirationChart({});
      return;
    }
    
    const chartData = analysisData.charts;

    // Respiração não tem toggle, está sempre 'live'
    renderRespirationChart(chartData);
  }


  // 6. Weekly Cough Chart
  function renderWeeklyCoughChart(chartData) {
    if (!chartData || !chartData.labels || !chartData.data) {
      showNoDataMessage('weeklyCoughChart');
      return;
    }
    const config = {
      type: 'bar',
      data: {
        labels: chartData.labels,
        datasets: [{
          label: 'Total de Tosses Detectadas',
          data: chartData.data,
          backgroundColor: 'rgba(255, 159, 64, 0.5)',
          borderColor: 'rgba(255, 159, 64, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, title: { display: true, text: 'Contagem Total de Tosses' } } },
        plugins: { legend: { display: false } }
      }
    };
    createOrUpdateChart('weeklyCoughChart', config);
  }

  // 8. Vitals History Chart (NOVO)
  function renderVitalsHistoryChart(chartData) {
    if (!chartData || !chartData.bpm || !chartData.spo2) {
      showNoDataMessage('vitalsHistoryChart');
      return;
    }
    const config = {
      type: 'line',
      data: {
        labels: chartData.labels,
        datasets: [
          {
            label: 'Batimentos (BPM)',
            data: chartData.bpm,
            borderColor: '#FF6384',
            yAxisID: 'yBPM',
            tension: 0.2,
            pointRadius: 1,
            spanGaps: true
          },
          {
            label: 'Saturação (SpO2)',
            data: chartData.spo2,
            borderColor: '#36A2EB',
            yAxisID: 'ySPO2',
            tension: 0.2,
            pointRadius: 1,
            spanGaps: true
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { title: { display: true, text: 'Última Hora' } },
          yBPM: { type: 'linear', position: 'left', title: { display: true, text: 'BPM' }, min: 40, max: 160 },
          ySPO2: { type: 'linear', position: 'right', title: { display: true, text: 'SpO2 (%)' }, min: 85, max: 101, grid: { drawOnChartArea: false } }
        },
        plugins: { legend: { position: 'bottom' }, tooltip: { mode: 'index', intersect: false } }
      }
    };
    createOrUpdateChart('vitalsHistoryChart', config);
  }

  // 9. Environment History Chart (NOVO)
  function renderEnvironmentHistoryChart(chartData) {
    if (!chartData || !chartData.temp || !chartData.humidity) {
      showNoDataMessage('environmentHistoryChart');
      return;
    }
    const config = {
      type: 'line',
      data: {
        labels: chartData.labels,
        datasets: [
          { label: 'Temperatura (°C)', data: chartData.temp, borderColor: '#FF9F40', yAxisID: 'yTemp', tension: 0.2, pointRadius: 1, spanGaps: true },
          { label: 'Umidade (%)', data: chartData.humidity, borderColor: '#4BC0C0', yAxisID: 'yHumidity', tension: 0.2, pointRadius: 1, spanGaps: true }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { title: { display: true, text: 'Última Hora' } },
          yTemp: { type: 'linear', position: 'left', title: { display: true, text: '°C' } },
          yHumidity: { type: 'linear', position: 'right', title: { display: true, text: '%' }, grid: { drawOnChartArea: false } }
        },
        plugins: { legend: { position: 'bottom' }, tooltip: { mode: 'index', intersect: false } }
      }
    };
    createOrUpdateChart('environmentHistoryChart', config);
  }

  function updateTriggers(triggers) {
    const triggersEl = document.getElementById('currentTriggers');
    if (!triggersEl) return;
    triggersEl.innerHTML = '';
    const list = triggers && triggers.length ? triggers : ['Nenhum gatilho crítico no momento.'];
    list.forEach(txt => {
      const li = document.createElement('li');
      li.textContent = txt;
      triggersEl.appendChild(li);
    });
  }

  // --- LÓGICA DE DADOS HISTÓRICOS ---
  
async function loadHistoricalData(chartId, sensorName, hours = 1) {
  const canvas = document.getElementById(chartId);
  if (!canvas) return;

  // Atualiza o texto do botão
  const button = document.querySelector(`.chart-toggle[data-chart-id="${chartId}"] .toggle-btn[data-mode="24h"]`);
  if (button) button.textContent = hours === 1 ? "Última 1h" : `Últimas ${hours}h`;

  showNoDataMessage(chartId, 'Carregando dados históricos...');

  try {
    const resp = await fetch(`/api/historical_data?sensor=${sensorName}&hours=${hours}`);
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { const j = await resp.json(); if (j?.error) msg = j.error; } catch {}
      throw new Error(msg);
    }
    const result = await resp.json();

    if (!result.points || !result.points.length) {
      showNoDataMessage(chartId, 'Sem dados históricos para este período.');
      return;
    }

    // ✨ Sanitiza (remove NaN/undef → null) e normaliza chaves
    const cleaned = sanitizePoints(result.points);
    if (!cleaned.length) {
      showNoDataMessage(chartId, 'Sem dados válidos para exibir.');
      return;
    }

    // Título do eixo Y padronizado a partir do mapa já existente
    const yTitle = yAxisLimits[chartId]?.title || 'Valor';
    // Label amigável (ex.: "Som", "Piezo", "Aceleração")
    const sensorLabel = (yAxisLimits[chartId]?.title.split('(')[0] || sensorName || 'Sinal').trim();

    const historicalConfig = {
      type: 'line',
      data: { datasets: [{
        label: `${sensorLabel} (Média ${hours}h)`,
        data: cleaned,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        spanGaps: true
      }]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'time',
            time: { tooltipFormat: 'dd/MM/yy HH:mm' },
            title: { display: true, text: 'Tempo' },
            ticks: { maxRotation: 0, autoSkip: true }
          },
          y: { title: { display: true, text: yTitle } }
        },
        plugins: { legend: { position: 'bottom' }, tooltip: { mode: 'index', intersect: false } }
      }
    };
    createOrUpdateChart(chartId, historicalConfig);

  } catch (err) {
    console.error(`Erro ao carregar dados históricos para ${chartId}:`, err);
    showNoDataMessage(chartId, `Falha: ${err.message}`);
  }
}

  /**
   * Cria um gráfico "live" vazio, pronto para receber dados via WebSocket ou polling.
   */
  function initLiveChart(chartId, label, yAxisTitle) {
    const config = {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: label,
          data: [],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1
        }]
      },
      options: {
        ...timeSeriesOptions,
        scales: { ...timeSeriesOptions.scales, y: { ...timeSeriesOptions.scales.y, title: { ...timeSeriesOptions.scales.y.title, text: yAxisTitle } } }
      }
    };
    createOrUpdateChart(chartId, config);
  }


  /**
   * Configura os botões de toggle para os gráficos de análise
   */
  function setupToggleButtons() {
    const toggles = document.querySelectorAll('.chart-toggle');
    
    toggles.forEach(toggle => {
      const chartId = toggle.dataset.chartId;
      const sensorName = toggle.dataset.sensorName;
      
      const buttons = toggle.querySelectorAll('.toggle-btn');
      
      buttons.forEach(button => {
        button.addEventListener('click', () => {
          // Lógica visual do botão
          buttons.forEach(btn => btn.classList.remove('active'));
          button.classList.add('active');

          const mode = button.dataset.mode;

          // Para todos os loops de atualização live antes de mudar de modo
          Object.keys(chartIntervals).forEach(stopLiveChartUpdate);

          // Lógica específica para o MPU6050
          // Este gráfico tem seu próprio loop de atualização e endpoint de histórico
          if (chartId === 'realtimeMotionChart') {
            stopMPUUpdates(); // Para o loop específico do MPU
            if (mode === 'live') {
              startMPUUpdates();
            } else if (mode === 'history') { // O botão agora é de histórico
              loadMPUHistory(60); // Carrega sempre a última 1 hora
            }
            return; // Impede que a lógica genérica abaixo seja executada para o MPU
          }

          // Lógica genérica para os outros gráficos de análise
          if (mode === 'live') {
            // Inicializa um gráfico live vazio e começa a buscar dados
            const yTitle = yAxisLimits[chartId]?.title || 'Valor';
            const label = (yAxisLimits[chartId]?.title.split('(')[0] || sensorName || 'Sinal').trim();
            initLiveChart(chartId, `Sinal de ${label}`, yTitle);
            startLiveChartUpdate(chartId, sensorName);

          } else if (mode === '24h') { // O botão de histórico para estes gráficos
            // O motion chart tem uma função de histórico especial
            if (chartId === 'motionChart') {
              loadMotionHistory(60); // Carrega 1 hora (60 min)
            } else {
              loadHistoricalData(chartId, sensorName, 1);
            }
          }
        });
      });
    });
  }


  // --- LÓGICA DO GRÁFICO MPU6050 ---
  let mpuUpdateInterval = null;
  const MPU_MAX_DATA_POINTS = 100; // Últimos 100 pontos

  function initMPUChart() {
    const canvas = document.getElementById('realtimeMotionChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const config = {
      type: 'line',
      data: {
        labels: [], // Timestamps
        datasets: [
          { label: 'Accel X', data: [], borderColor: '#FF6384', tension: 0.1, pointRadius: 0 },
          { label: 'Accel Y', data: [], borderColor: '#36A2EB', tension: 0.1, pointRadius: 0 },
          { label: 'Accel Z', data: [], borderColor: '#4BC0C0', tension: 0.1, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'second', displayFormats: { second: 'HH:mm:ss' } }, ticks: { maxRotation: 0, autoSkip: true } },
          y: { title: { display: true, text: 'Aceleração (m/s^2)' }, min: -20, max: 20 } // Limite fixo
        },
        animation: false // Desativa animação para fluidez
      }
    };
    createOrUpdateChart('realtimeMotionChart', config);
  }

  async function fetchMPULatest() {
    const mpuChart = charts['realtimeMotionChart'];
    if (!mpuChart) return;

    try {
      const res = await fetch('/api/mpu6050_latest');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      const now = new Date(data.timestamp);
      mpuChart.data.labels.push(now);
      mpuChart.data.datasets[0].data.push(data.accel_x);
      mpuChart.data.datasets[1].data.push(data.accel_y);
      mpuChart.data.datasets[2].data.push(data.accel_z);

      while (mpuChart.data.labels.length > MPU_MAX_DATA_POINTS) {
        mpuChart.data.labels.shift();
        mpuChart.data.datasets.forEach(ds => ds.data.shift());
      }
      mpuChart.update('none');

      // Atualiza o indicador de estado
      const stateEl = document.getElementById('userStateText');
      if (stateEl) {
        const states = ['Parado', 'Andando', 'Corrida Leve', 'Corrida Rápida', 'Queda Detectada'];
        const stateIndex = parseInt(data.state, 10);
        stateEl.textContent = `Estado: ${states[stateIndex] || 'Indefinido'}`;
      }

    } catch (e) {
      console.error("Falha ao buscar dados do MPU:", e);
      const stateEl = document.getElementById('userStateText');
      if (stateEl) stateEl.textContent = 'Estado: Desconectado';
      stopMPUUpdates(); // Para o loop se houver erro
    }
  }

  async function loadMPUHistory(minutes) {
    const canvas = document.getElementById('realtimeMotionChart');
    if (!canvas) return;
    showNoDataMessage('realtimeMotionChart', 'Carregando histórico MPU...');

    try {
      const res = await fetch(`/api/mpu6050_history?minutes=${minutes}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      if (!data.points || data.points.length === 0) {
        showNoDataMessage('realtimeMotionChart', 'Sem dados históricos para este período.');
        return;
      }

      const config = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Accel X (Média)', data: data.points.map(p => ({ x: p.t, y: p.ax })), borderColor: '#FF6384', tension: 0.1, pointRadius: 0, spanGaps: true },
            { label: 'Accel Y (Média)', data: data.points.map(p => ({ x: p.t, y: p.ay })), borderColor: '#36A2EB', tension: 0.1, pointRadius: 0, spanGaps: true },
            { label: 'Accel Z (Média)', data: data.points.map(p => ({ x: p.t, y: p.az })), borderColor: '#4BC0C0', tension: 0.1, pointRadius: 0, spanGaps: true }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { 
              type: 'time', 
              time: { tooltipFormat: 'dd/MM HH:mm', unit: 'minute', displayFormats: { minute: 'HH:mm' } }, 
              title: { display: true, text: 'Tempo (Última Hora)' } 
            },
            y: { title: { display: true, text: 'Aceleração Média (m/s^2)' } }
          },
          plugins: { tooltip: { mode: 'index', intersect: false } }
        }
      };
      createOrUpdateChart('realtimeMotionChart', config);
    } catch (e) {
      console.error("Falha ao carregar histórico MPU:", e);
      showNoDataMessage('realtimeMotionChart', `Falha: ${e.message}`);
    }
  }

  function startMPUUpdates() {
    if (mpuUpdateInterval) clearInterval(mpuUpdateInterval);
    
    initMPUChart(); // Inicializa o gráfico limpo
    fetchMPULatest(); // Busca o primeiro ponto
    // O .ino envia a 1Hz (1000ms)
    mpuUpdateInterval = setInterval(fetchMPULatest, 1000); 
    document.getElementById('userStateIndicator').style.display = 'block';
  }

  function stopMPUUpdates() {
    if (mpuUpdateInterval) {
      clearInterval(mpuUpdateInterval);
      mpuUpdateInterval = null;
    }
    document.getElementById('userStateIndicator').style.display = 'none';
  }

  async function loadMotionHistory(minutes = 60) {
  const chartId = 'motionChart';
  showNoDataMessage(chartId, 'Carregando histórico de movimento...');

  try {
    const res = await fetch(`/api/motion_history?minutes=${minutes}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (!data.points || !data.points.length) {
      showNoDataMessage(chartId, 'Sem dados históricos nesse período.');
      return;
    }

    const dsAcc = data.points.map(p => ({ x: p.t, y: Number(p.acc) }));
    const dsGyro = data.points.map(p => ({ x: p.t, y: Number(p.gyro) }));

    const cfg = {
      type: 'line',
      data: {
        datasets: [
          { label: 'Intensidade do Movimento (Acel.)', data: dsAcc, borderColor: '#6a5acd', backgroundColor: 'rgba(106,90,205,0.12)', fill: true, pointRadius: 0, tension: 0.25, spanGaps: true },
          { label: 'Intensidade da Rotação (Giro.)', data: dsGyro, borderColor: '#00a896', pointRadius: 0, tension: 0.25, spanGaps: true }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { 
            type: 'time', 
            time: { tooltipFormat: 'dd/MM HH:mm', unit: 'minute', displayFormats: { minute: 'HH:mm' } }, 
            title: { display: true, text: 'Tempo (Última Hora)' }, ticks: { maxRotation: 0, autoSkip: true } 
          },
          y: { title: { display: true, text: 'Magnitude (Acel. e Rotação)' } }
        },
        plugins: { tooltip: { mode: 'index', intersect: false }, legend: { position: 'bottom' } }
      }
    };
    createOrUpdateChart(chartId, cfg);
  } catch (e) {
    console.error('Erro ao carregar histórico de movimento:', e);
    showNoDataMessage(chartId, `Falha: ${e.message}`);
  }
}

  // --- LÓGICA DO CALENDÁRIO ---
  
  function handleDateClick(info) {
    const date = info.dateStr;
    const existing = calendar.getEvents().filter(e => e.startStr === date);
    if (existing.length) {
      if (confirm('Deseja remover este registro de uso?')) {
        fetch(`/api/events?id=${existing[0].id}`, { method: 'DELETE' })
          .then(resp => { if (resp.ok) existing[0].remove(); });
      }
    } else {
      fetch('/api/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date })
      })
        .then(r => r.json())
        .then(evt => calendar.addEvent(evt));
    }
  }

  function initializeCalendar(events) {
    const calendarEl = document.getElementById('inhalerCalendar');
    if (calendar || !calendarEl) return;
    calendar = new FullCalendar.Calendar(calendarEl, {
      initialView: 'dayGridMonth',
      locale: 'pt-br',
      height: 'auto',
      contentHeight: 350,
      events: events,
      dateClick: handleDateClick,
      eventClick: (info) => {
        if (confirm('Deseja remover este registro de uso?')) {
          fetch(`/api/events?id=${info.event.id}`, { method: 'DELETE' })
            .then(resp => { if (resp.ok) info.event.remove(); });
        }
      }
    });
    calendar.render();
  }

  // --- FUNÇÃO PRINCIPAL DE CARREGAMENTO DE DADOS ---

  function loadData() {
    if (isLoadingData) {
      console.log('Aguardando a requisicao anterior...');
      return;
    }
    isLoadingData = true;

    fetch('/api/data')
      .then(response => {
        if (!response.ok) throw new Error(`Falha na API: ${response.statusText}`);
        return response.json();
      })
      .then(data => {
        const envData = data.env_data;

        if (data.error) {
          console.error('Erro nos dados recebidos do servidor:', data.error);
          updateTriggers([`Erro ao processar dados: ${data.error}`]);
        }

        if (envData && !envData.error) {
          updateResults(envData.analysis.results);
          updateTriggers(envData.triggers);
          updatePerfGauge(envData.pefr_prediction); // Atualiza o PERF
          
          // O loop principal agora só atualiza o gráfico de respiração
          renderRespirationChart(envData.analysis.charts);
          
          // Gráfico de tosse é sempre atualizado
          renderWeeklyCoughChart(envData.weekly_cough_data);

          // Renderiza os novos gráficos
          renderVitalsHistoryChart(envData.vitals_history_chart);
          renderEnvironmentHistoryChart(envData.env_history_chart);

          const timestampEl = document.getElementById('lastUpdated');
          if (timestampEl && envData.timestamp) {
            const formatted = new Date(envData.timestamp).toLocaleTimeString('pt-BR');
            timestampEl.textContent = `Ultima atualizacao: ${formatted}`;
          }
        } else {
          // Lida com erros ou dados vazios
          updateResults({ 'batimentos-cardiacos': 0, 'saturacao': 0, 'temperatura-corporal': 0, 'som': 0 });
          renderRespirationChart(null); // Limpa gráfico de respiração
          renderWeeklyCoughChart(null);
          updatePerfGauge(null); // Limpa PERF
          renderVitalsHistoryChart(null);
          renderEnvironmentHistoryChart(null);
          if (!data.error) {
            updateTriggers([envData?.error || 'Dados ambientais indisponiveis no momento.']);
          }
        }

        const usageEvents = data.usage_events || [];
        if (calendar) {
          calendar.removeAllEvents();
          calendar.addEventSource(usageEvents);
        } else {
          initializeCalendar(usageEvents);
        }
      })
      .catch(error => {
        console.error('Falha ao carregar dados:', error);
        // Limpa todos os gráficos em caso de falha de conexão
        Object.keys(charts).forEach(key => {
            if(key !== 'realtimeMotionChart') showNoDataMessage(key, 'Erro de conexão');
        });
        updateTriggers([`Erro de conexao: ${error.message}`]);
        updatePerfGauge(null);
      })
      .finally(() => {
        isLoadingData = false;
      });
  }
  
  // --- INICIALIZAÇÃO DA PÁGINA ---

  // Renderiza os gráficos de ANÁLISE (que são mais complexos) com os dados embutidos na página
  if (window.env_data && window.env_data.analysis && window.env_data.analysis.charts) {
    const initialChartData = window.env_data.analysis.charts;
    renderRespirationChart(initialChartData);
  } else {
    // Mostra mensagem se não houver dados iniciais
    showNoDataMessage('respirationChart');
  }

  // Inicializa os gráficos "live" simples (PPG, Som, Movimento, Piezo)
  initLiveChart('ppgChart', 'Sinal PPG', yAxisLimits.ppgChart.title);
  initLiveChart('soundChart', 'Sinal de Som', yAxisLimits.soundChart.title);
  initLiveChart('motionChart', 'Intensidade do Movimento', yAxisLimits.motionChart.title);
  initLiveChart('piezoChart', 'Sinal Piezo', yAxisLimits.piezoChart.title);

  if(window.env_data && window.env_data.weekly_cough_data) {
    renderWeeklyCoughChart(window.env_data.weekly_cough_data);
  } else {
    showNoDataMessage('weeklyCoughChart');
  }

  renderVitalsHistoryChart(window.env_data ? window.env_data.vitals_history_chart : null);
  renderEnvironmentHistoryChart(window.env_data ? window.env_data.env_history_chart : null);

  // Inicializa o medidor PERF com dados embutidos
  updatePerfGauge(window.env_data ? window.env_data.pefr_prediction : null);

  // Inicializa o calendário com dados embutidos
  initializeCalendar(window.usage_events || []);
  
  // Configura os botões de toggle (Live/Histórico)
  setupToggleButtons();
  
  // Inicia os gráficos de análise no modo "live" por padrão
  startLiveChartUpdate('ppgChart', 'batimentos-cardiacos');
  startLiveChartUpdate('soundChart', 'som');
  startLiveChartUpdate('motionChart', 'acelerometro-z'); // Usando um sensor representativo
  startLiveChartUpdate('piezoChart', 'piezo');
  
  // Inicia o gráfico MPU6050 também no modo "live"
  startMPUUpdates();

  // Inicia o loop de atualização principal (que agora atualiza PERF, cards e gráficos 'live')
  loadData(); // Carga imediata
  setInterval(loadData, 2000); // Atualiza a cada 2 segundos

  // Lógica da seção de gráficos recolhível (sem alterações)
  const chartsSection = document.querySelector('.charts-section');
  const toggleBtn = document.querySelector('.toggle-charts-btn');
  const header = document.querySelector('.charts-header');
  function toggleCharts(e) {
    e.stopPropagation();
    if(!chartsSection) return;
    const isCollapsed = chartsSection.classList.toggle('collapsed');
    if(toggleBtn) toggleBtn.setAttribute('aria-expanded', !isCollapsed);
  }
  if(toggleBtn) toggleBtn.addEventListener('click', toggleCharts);
  if(header) header.addEventListener('click', (e) => {
    if (e.target !== toggleBtn && !toggleBtn.contains(e.target)) {
      toggleCharts(e);
    }
  });

  // Aplica layout de grade (sem alterações)
  try {
    const chartsContainer = document.querySelector('.charts-container');
    if (chartsContainer) {
      chartsContainer.style.display = 'grid';
      chartsContainer.style.gridTemplateColumns = '1fr 1fr';
      chartsContainer.style.gap = '20px';
    }
  } catch (e) {
    console.error("Falha ao tentar aplicar o layout de grade aos gráficos.", e);
  }
});
