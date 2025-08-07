// insights.js

document.addEventListener('DOMContentLoaded', () => {
  let calendar;
  const charts = {
    ppgChart: null,
    respirationChart: null,
    soundChart: null,
    motionChart: null
  };

  // --- MINIGAME LOGIC ---
  const breathingText = document.getElementById("breathingText"),
        breathingCircle = document.getElementById("breathingCircle"),
        circleProgress = document.querySelector(".circle-progress"),
        startButton = document.getElementById("startGameBtn"),
        downloadContainer = document.getElementById("downloadContainer"),
        // Adicionado para exibir o resultado da predição
        predictionResultEl = document.getElementById("predictionResult"),
        radius = 65,
        circumference = 2 * Math.PI * radius;
  circleProgress.style.strokeDasharray = `${circumference}`;
  circleProgress.style.strokeDashoffset = `${circumference}`;
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
      
      // --- INÍCIO: LÓGICA MODIFICADA DE UPLOAD E PREDIÇÃO ---
      mediaRecorder.onstop = () => {
        let blob = new Blob(audioChunks, { type: "audio/webm" }),
            url = URL.createObjectURL(blob);
        downloadContainer.innerHTML = "";
        let a = document.createElement("a");
        a.href = url;
        a.download = "gravacao_respiracao.webm";
        a.textContent = "";    //BOTAO DOWNLOAD TEXTO  "⬇️ Baixar áudio da respiração"
        a.className = "download-btn";
        downloadContainer.appendChild(a);

        // 1. Mostrar status de análise e limpar resultados antigos
        if (predictionResultEl) {
            predictionResultEl.innerHTML = '<p class="loading">Analisando o áudio...</p>';
            predictionResultEl.style.display = 'block';
        }

        let formData = new FormData();
        formData.append("audio", blob, "gravacao_respiracao.webm");

        // 2. Fazer upload do áudio para conversão
        fetch("/api/upload-audio", { method: "POST", body: formData })
          .then(res => {
              if (!res.ok) throw new Error(`Falha no upload (${res.status})`);
              return res.json();
          })
          .then(uploadData => {
              console.log("✅ Upload concluído:", uploadData);
              if (uploadData.error) throw new Error(uploadData.error);
              
              // 3. Chamar a API de predição com o caminho do arquivo .wav
              return fetch("/api/predict-audio", {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ wav_path: uploadData.wav_path })
              });
          })
          .then(res => {
              if (!res.ok) throw new Error(`Falha na predição (${res.status})`);
              return res.json();
          })
          .then(predictionData => {
              console.log("🧠 Predição recebida:", predictionData);
              // 4. Exibir o resultado da predição na tela
              if (predictionResultEl) {
                  if (predictionData.error) {
                      predictionResultEl.innerHTML = `<p class="error">Erro na Análise: ${predictionData.error}</p>`;
                  } else {
                      const confidencePercent = (predictionData.confidence * 100).toFixed(1);
                      predictionResultEl.innerHTML = `
                          <p class="result-title">Resultado da Análise:</p>
                          <p class="result-class">${predictionData.predicted_class}</p>
                          <p class="result-confidence">Confiança: ${confidencePercent}%</p>
                      `;
                  }
              }
          })
          .catch(err => {
              console.error("❌ Erro no processo de análise de áudio:", err);
              if (predictionResultEl) {
                  predictionResultEl.innerHTML = `<p class="error">Falha ao processar o áudio: ${err.message}</p>`;
              }
          });
      };
      // --- FIM: LÓGICA MODIFICADA DE UPLOAD E PREDIÇÃO ---
      
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
      // Limpar resultados anteriores ao iniciar
      if (predictionResultEl) predictionResultEl.style.display = 'none';
      if (downloadContainer) downloadContainer.innerHTML = "";
      
      await iniciarGravacao();
      isRunning = true;
      phaseIndex = 0;
      animatePhase(phases[phaseIndex]);
      startButton.textContent = "Parar Exercício";
    }
  });

  /**
   * Updates the main result cards with sensor data.
   */
  function updateResults(results) {
    const container = document.getElementById('analysisResults');
    container.innerHTML = '';
    const metrics = [
      { label: 'Batimentos Cardíacos', value: results['batimentos-cardiacos'] > 0 ? results['batimentos-cardiacos'].toFixed(0) : 'N/A', unit: 'BPM' },
      { label: 'Saturação', value: results['saturacao'] > 0 ? results['saturacao'].toFixed(1) : 'N/A', unit: '%' },
      { label: 'Temperatura Corporal', value: results['temperatura-corporal'] > 0 ? results['temperatura-corporal'].toFixed(2) : 'N/A', unit: '°C' },
      { label: 'Temperatura Oximetro', value: results['temperatura-oximetro'] > 0 ? results['temperatura-oximetro'].toFixed(2) : 'N/A', unit: '°C' },
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
   * Generic function to create or update a chart.
   */
  function createOrUpdateChart(chartId, config) {
    const ctx = document.getElementById(chartId)?.getContext('2d');
    if (!ctx) {
      console.warn(`Canvas com ID ${chartId} não encontrado.`);
      return;
    }
    if (charts[chartId]) {
      charts[chartId].destroy();
    }
    charts[chartId] = new Chart(ctx, config);
  }

  /**
   * Helper function to display a "No Data" message on a canvas.
   */
  function showNoDataMessage(chartId) {
    const canvas = document.getElementById(chartId);
    if (!canvas) {
      console.warn(`Canvas com ID ${chartId} não encontrado.`);
      return;
    }
    const ctx = canvas.getContext('2d');
    if (charts[chartId]) {
      charts[chartId].destroy();
      charts[chartId] = null;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#999';
    ctx.font = '16px sans-serif';
    ctx.fillText('Dados insuficientes para exibir o gráfico', canvas.width / 2, canvas.height / 2);
    ctx.restore();
  }

  /**
   * Renders all analysis charts, with checks for data availability and timestamp.
   */
  function renderAnalysisCharts(analysisData, timestamp) {
    if (!analysisData || !analysisData.charts) {
      Object.keys(charts).forEach(showNoDataMessage);
      return;
    }

    const chartData = analysisData.charts;
    const commonOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
    };

    // 1. PPG Chart
    if (chartData.ppg?.signal?.length > 0 && chartData.time_axis?.length > 0) {
      const ppgSignalData = chartData.time_axis.map((t, i) => ({
        x: t,
        y: chartData.ppg.signal[i] || 0
      }));
      const ppgPeaksData = chartData.ppg.peaks_time.map((t, i) => ({
        x: t,
        y: chartData.ppg.peaks_value[i] || 0
      }));
      const ppgConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: `Sinal PPG Filtrado`, data: ppgSignalData, borderColor: '#FF6F61', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            { type: 'scatter', label: `Batimentos Detectados}`, data: ppgPeaksData, backgroundColor: '#D32F2F', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: {
          ...commonOptions,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Tempo (s)' } },
            y: { title: { display: true, text: 'Amplitude Filtrada' } }
          }
        }
      };
      createOrUpdateChart('ppgChart', ppgConfig);
    } else {
      showNoDataMessage('ppgChart');
    }

    // 2. Respiration Chart
    if (chartData.respiration?.signal?.length > 0 && chartData.respiration?.time_axis?.length > 0) {
      const respSignalData = chartData.respiration.time_axis.map((t, i) => ({
        x: t,
        y: chartData.respiration.signal[i] || 0
      }));
      const respPeaksData = chartData.respiration.peaks_time.map((t, i) => ({
        x: t,
        y: chartData.respiration.peaks_value[i] || 0
      }));
      const respConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: `Sinal Respiratório`, data: respSignalData, borderColor: '#2196F3', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            { type: 'scatter', label: `Picos Respiratórios`, data: respPeaksData, backgroundColor: '#1976D2', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: {
          ...commonOptions,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Tempo (s)' } },
            y: { title: { display: true, text: 'Amplitude Modulada' } }
          }
        }
      };
      createOrUpdateChart('respirationChart', respConfig);
    } else {
      showNoDataMessage('respirationChart');
    }

    // 3. Sound Chart
    if (chartData.sound?.signal?.length > 0 && chartData.time_axis?.length > 0) {
      const soundSignalData = chartData.time_axis.map((t, i) => ({
        x: t,
        y: chartData.sound.signal[i] || 0
      }));
      const soundThresholdData = chartData.time_axis.map(t => ({
        x: t,
        y: chartData.sound.threshold || 0
      }));
      const soundPeaksData = chartData.sound.peaks_time.map((t, i) => ({
        x: t,
        y: chartData.sound.peaks_value[i] || 0
      }));
      const soundConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: `Sinal do Microfone`, data: soundSignalData, borderColor: '#4CAF50', borderWidth: 1.5, pointRadius: 0 },
            { label: `Limiar de Som`, data: soundThresholdData, borderColor: '#F44336', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: `Picos de Som`, data: soundPeaksData, backgroundColor: '#2E7D32', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: {
          ...commonOptions,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Tempo (s)' } },
            y: { title: { display: true, text: 'Amplitude do ADC' } }
          }
        }
      };
      createOrUpdateChart('soundChart', soundConfig);
    } else {
      showNoDataMessage('soundChart');
    }

    // 4. Motion Chart
    if (chartData.motion?.signal?.length > 0 && chartData.time_axis?.length > 0) {
      const motionSignalData = chartData.time_axis.map((t, i) => ({
        x: t,
        y: chartData.motion.signal[i] || 0
      }));
      const motionThresholdData = chartData.time_axis.map(t => ({
        x: t,
        y: chartData.motion.threshold || 0
      }));
      const motionPeaksData = chartData.motion.peaks_time.map((t, i) => ({
        x: t,
        y: chartData.motion.peaks_value[i] || 0
      }));
      const motionConfig = {
        type: 'line',
        data: {
          datasets: [
            { label: `Magnitude da Aceleração`, data: motionSignalData, borderColor: '#9C27B0', borderWidth: 1.5, pointRadius: 0 },
            { label: `Limiar de Movimento`, data: motionThresholdData, borderColor: '#FF9800', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 },
            { type: 'scatter', label: `Solavancos Detectados`, data: motionPeaksData, backgroundColor: '#F57C00', pointStyle: 'crossRot', radius: 6, borderWidth: 2 }
          ]
        },
        options: {
          ...commonOptions,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Tempo (s)' } },
            y: { title: { display: true, text: 'Magnitude (g)' } }
          }
        }
      };
      createOrUpdateChart('motionChart', motionConfig);
    } else {
      showNoDataMessage('motionChart');
    }
  }

    // 5. Weekly Cough Chart
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
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Contagem Total de Tosses'
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    };
    createOrUpdateChart('weeklyCoughChart', config);
  }

  function updateTriggers(triggers) {
    const triggersEl = document.getElementById('currentTriggers');
    triggersEl.innerHTML = '';
    const list = triggers && triggers.length ? triggers : ['Nenhum gatilho crítico no momento.'];
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
    if (calendar) return;
    const calendarEl = document.getElementById('inhalerCalendar');
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
    console.log("🔄 Carregando novos dados...");
    fetch('/api/data')
      .then(r => {
        if (!r.ok) throw new Error(`Falha na API: ${r.statusText}`);
        return r.json();
      })
      .then(data => {
        console.log("✅ Dados recebidos:", data);
        if (data.env_data && !data.env_data.error) {
          updateResults(data.env_data.analysis.results);
          renderAnalysisCharts(data.env_data.analysis, data.env_data.timestamp);
          updateTriggers(data.env_data.triggers);
          // --- MODIFICAÇÃO: Chamar a função para renderizar o novo gráfico ---
          renderWeeklyCoughChart(data.env_data.weekly_cough_data); 
        } else {
          console.warn("Dados de análise não encontrados ou erro na API:", data.env_data?.error || 'Desconhecido');
          updateResults({
            // ... (objeto de resultados zerado, sem alterações)
          });
          renderAnalysisCharts(null, null);
          // --- MODIFICAÇÃO: Mostrar mensagem de "sem dados" no novo gráfico também ---
          renderWeeklyCoughChart(null); 
          updateTriggers([data.env_data?.error || 'Erro ao carregar dados.']);
        }
        initializeCalendar(data.usage_events || []);
        if (calendar) {
          calendar.removeAllEvents();
          calendar.addEventSource(data.usage_events || []);
        }
      })
      .catch(err => {
        console.error("❌ Erro ao carregar dados:", err);
        updateResults({
          // ... (objeto de resultados zerado, sem alterações)
        });
        renderAnalysisCharts(null, null);
        // --- MODIFICAÇÃO: Mostrar mensagem de "sem dados" no novo gráfico em caso de erro ---
        renderWeeklyCoughChart(null);
        updateTriggers([`Erro de conexão: ${err.message}`]);
      });
  }

  // --- ALTERAÇÃO 1: APLICAR LAYOUT 2x2 AOS GRÁFICOS ---
  try {
    const firstChart = document.getElementById('ppgChart');
    if (firstChart && firstChart.parentElement && firstChart.parentElement.parentElement) {
      const chartsContainer = firstChart.parentElement.parentElement;
      chartsContainer.style.display = 'grid';
      chartsContainer.style.gridTemplateColumns = '1fr 1fr';
      chartsContainer.style.gap = '20px';
    }
  } catch (e) {
    console.error("Falha ao tentar aplicar o layout de grade aos gráficos.", e);
  }

  // Inicia o carregamento de dados
  loadData();
  // --- ALTERAÇÃO 2: ATUALIZAR DADOS A CADA 5 SEGUNDOS ---
  setInterval(loadData, 5000);

  // Lógica da seção de gráficos recolhível (sem alteração)
  const chartsSection = document.querySelector('.charts-section');
  const toggleBtn = document.querySelector('.toggle-charts-btn');
  const header = document.querySelector('.charts-header');
  function toggleCharts(e) {
    e.stopPropagation();
    const isCollapsed = chartsSection.classList.toggle('collapsed');
    toggleBtn.setAttribute('aria-expanded', !isCollapsed);
  }
  toggleBtn.addEventListener('click', toggleCharts);
  header.addEventListener('click', (e) => {
    if (e.target !== toggleBtn && !toggleBtn.contains(e.target)) {
      toggleCharts(e);
    }
  });
});