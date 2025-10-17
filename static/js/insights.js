document.addEventListener('DOMContentLoaded', () => {
  // Inicializa um objeto para manter as instÃ¢ncias dos grÃ¡ficos
  let calendar;
  const charts = {
    ppgChart: null,
    respirationChart: null,
    soundChart: null,
    motionChart: null,
    piezoChart: null,
    weeklyCoughChart: null
  };
  let isLoadingData = false; // Variavel de controle para evitar race conditions

  // --- MINIGAME LOGIC (sem alteraÃ§Ãµes) ---
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
            predictionResultEl.innerHTML = '<p class="loading">Analisando o Ã¡udio...</p>';
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
              if (!ok) throw new Error(data.error || `Falha na prediÃ§Ã£o (${status})`);
              if (predictionResultEl) {
                  const confidencePercent = (data.confidence * 100).toFixed(1);
                  predictionResultEl.innerHTML = `
                      <p class="result-title">Resultado da AnÃ¡lise:</p>
                      <p class="result-class">${data.predicted_class}</p>
                      <p class="result-confidence">ConfianÃ§a: ${confidencePercent}%</p>
                  `;
              }
          })
          .catch(err => {
              console.error("âŒ Erro no processo de anÃ¡lise de Ã¡udio:", err);
              if (predictionResultEl) {
                  predictionResultEl.innerHTML = `<p class="error">Falha ao processar o Ã¡udio: ${err.message}</p>`;
              }
          });
      };
      mediaRecorder.start();
    } catch (e) {
      console.error("Erro ao obter mÃ­dia:", e);
      alert("NÃ£o foi possÃ­vel acessar o microfone. Verifique as permissÃµes do navegador.");
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
        startButton.textContent = "Iniciar ExercÃ­cio";
        pararGravacao();
      } else {
        if (predictionResultEl) predictionResultEl.style.display = 'none';
        if (downloadContainer) downloadContainer.innerHTML = "";
        
        await iniciarGravacao();
        isRunning = true;
        phaseIndex = 0;
        animatePhase(phases[phaseIndex]);
        startButton.textContent = "Parar ExercÃ­cio";
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
      { label: 'Batimentos Cardí­acos', value: results['batimentos-cardiacos'] > 0 ? results['batimentos-cardiacos'].toFixed(0) : 'N/A', unit: 'BPM' },
      { label: 'Saturação', value: results['saturacao'] > 0 ? results['saturacao'].toFixed(1) : 'N/A', unit: '%' },
      { label: 'Temperatura Corporal', value: results['temperatura-corporal'] > 0 ? results['temperatura-corporal'].toFixed(2) : 'N/A', unit: '°C' },
      { label: 'Ní­vel de Ruí­do', value: results['som'] > 0 ? results['som'].toFixed(0) : 'N/A', unit: 'ADC' },
    ];
    metrics.forEach(metric => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.innerHTML = `<p class="value">${metric.value} <span style="font-size: 1.2rem;">${metric.unit}</span></p><p class="label">${metric.label}</p>`;
      container.appendChild(card);
    });
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
  function showNoDataMessage(chartId) {
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
    ctx.fillText('Dados insuficientes para exibir o grÃ¡fico', canvas.width / 2, canvas.height / 2);
    ctx.restore();
}

  /**
   * Converts a timestamp to a relative time string (e.g., "X seconds ago").
   */
  function formatRelativeTime(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diffSeconds = (now - time) / 1000;
    if (diffSeconds < 60) return `${Math.round(diffSeconds)}s atrÃ¡s`;
    return time.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  /**
   * Renders all analysis charts with improved time axis and fixed Y-axis limits.
   */
  function renderAnalysisCharts(analysisData) {
    if (!analysisData || !analysisData.charts) {
      Object.keys(charts).forEach(key => {
        if (key !== 'weeklyCoughChart') showNoDataMessage(key);
      });
      return;
    }

    const chartData = analysisData.charts;
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
      motionChart: { min: 0, max: 60, title: 'Magnitude (Acel. e RotaÃ§Ã£o)' },
      piezoChart: { min: 0, max: 5000, title: 'Amplitude do Sinal' }
    };

    // 1. PPG Chart
    if (chartData.ppg?.signal?.length > 1) {
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

    // 2. Respiration Chart
    if (chartData.respiration?.signal?.length > 1) {
      const respConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal RespiratÃ³rio Derivado', data: chartData.respiration.signal, borderColor: '#2196F3', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            { type: 'scatter', label: 'Picos RespiratÃ³rios', data: chartData.respiration.peaks, backgroundColor: '#1976D2', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
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

    // 3. Sound Chart
    if (chartData.sound?.signal?.length > 1) {
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

    // 4. Motion Chart
    if (chartData.motion?.signal?.length > 1) {
      const motionConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Intensidade do Movimento (Acel.)', data: chartData.motion.signal, borderColor: '#6a5acd', backgroundColor: 'rgba(106, 90, 205, 0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 },
            { label: 'Intensidade da RotaÃ§Ã£o (Giro.)', data: chartData.motion.rotation_signal, borderColor: '#00a896', fill: false, borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Limiar de DetecÃ§Ã£o de Tosse', data: chartData.motion.signal.map(p => ({ x: p.x, y: chartData.motion.threshold })), borderColor: '#aaaaaa', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
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

    // 5. Piezo Chart
    if (chartData.piezo?.signal?.length > 1) {
      const piezoConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: 'Sinal Piezo (Mov. TorÃ¡cico)', data: chartData.piezo.signal, borderColor: '#FF9800', borderWidth: 1.5, pointRadius: 0 },
            { label: 'Limiar de VibraÃ§Ã£o', data: chartData.piezo.signal.map(p => ({ x: p.x, y: chartData.piezo.threshold })), borderColor: '#F44336', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: 'Picos de VibraÃ§Ã£o', data: chartData.piezo.peaks, backgroundColor: '#EF6C00', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
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

  function updateTriggers(triggers) {
    const triggersEl = document.getElementById('currentTriggers');
    if (!triggersEl) return;
    triggersEl.innerHTML = '';
    const list = triggers && triggers.length ? triggers : ['Nenhum gatilho crí­tico no momento.'];
    list.forEach(txt => {
      const li = document.createElement('li');
      li.textContent = txt;
      triggersEl.appendChild(li);
    });
  }

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
          renderAnalysisCharts(envData.analysis);
          renderWeeklyCoughChart(envData.weekly_cough_data);

          const timestampEl = document.getElementById('lastUpdated');
          if (timestampEl && envData.timestamp) {
            const formatted = new Date(envData.timestamp).toLocaleTimeString('pt-BR');
            timestampEl.textContent = `Ultima atualizacao: ${formatted}`;
          }
        } else {
          updateResults({ 'batimentos-cardiacos': 0, 'saturacao': 0, 'temperatura-corporal': 0, 'som': 0 });
          renderAnalysisCharts(null);
          renderWeeklyCoughChart(null);
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
        const allChartIds = Object.keys(charts);
        allChartIds.forEach(showNoDataMessage);
        updateTriggers([`Erro de conexao: ${error.message}`]);
      })
      .finally(() => {
        isLoadingData = false;
      });
  }
  // Aplica layout de grade aos grÃ¡ficos
  try {
    const chartsContainer = document.querySelector('.charts-container');
    if (chartsContainer) {
      chartsContainer.style.display = 'grid';
      chartsContainer.style.gridTemplateColumns = '1fr 1fr';
      chartsContainer.style.gap = '20px';
    }
  } catch (e) {
    console.error("Falha ao tentar aplicar o layout de grade aos grÃ¡ficos.", e);
  }

  // Inicia o carregamento de dados e define a atualizaÃ§Ã£o periÃ³dica
  loadData();
  setInterval(loadData, 2000);

  // LÃ³gica da seÃ§Ã£o de grÃ¡ficos recolhÃ­vel
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
});

