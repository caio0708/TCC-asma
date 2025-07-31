document.addEventListener('DOMContentLoaded', () => {
  let calendar;
  const charts = {};

const breathingText = document.getElementById('breathingText');
const breathingCircle = document.getElementById('breathingCircle');
const circleProgress = document.querySelector('.circle-progress');
const startButton = document.getElementById('startGameBtn');
const downloadContainer = document.getElementById('downloadContainer');

const radius = 65;
const circumference = 2 * Math.PI * radius;

circleProgress.style.strokeDasharray = `${circumference}`;
circleProgress.style.strokeDashoffset = `${circumference}`;

const phases = [
  { name: 'inhale', duration: 2000, scale: 1.3, text: 'Inspire' },
  { name: 'hold', duration: 1000, scale: 1.3, text: 'Prenda' },
  { name: 'exhale', duration: 3000, scale: 0.7, text: 'Expire' },
  { name: 'hold', duration: 1000, scale: 0.7, text: 'Prenda' },
];

let phaseIndex = 0;
let isRunning = false;
let phaseTimeout = null;

// 🎙️ Variáveis da gravação
let mediaRecorder;
let audioChunks = [];

// 🎧 Função para iniciar a gravação
async function iniciarGravacao() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  audioChunks = [];

  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    const url = URL.createObjectURL(blob);

    // Limpa área de download
    downloadContainer.innerHTML = '';

    // Cria botão para baixar áudio
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gravacao_respiracao.webm';
    a.textContent = '⬇️ Baixar áudio da respiração';
    a.className = 'download-btn';
    downloadContainer.appendChild(a);

    // Envia áudio para backend Flask
    const formData = new FormData();
    formData.append('audio', blob, 'gravacao_respiracao.webm');

    fetch('/api/upload-audio', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        console.log('✅ Upload concluído:', data);
        const info = document.createElement('p');
        info.textContent = `Áudio processado: ${data.wav_path || 'Indefinido'}`;
        downloadContainer.appendChild(info);
      })
      .catch(err => {
        console.error('Erro no upload:', err);
        const error = document.createElement('p');
        error.style.color = 'red';
        error.textContent = '❌ Erro ao enviar áudio para o servidor.';
        downloadContainer.appendChild(error);
      });
  };

  mediaRecorder.start();
  console.log('🎙️ Gravação iniciada');
}

// 🛑 Função para parar gravação
function pararGravacao() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    console.log('🛑 Gravação encerrada');
  }
}

function animatePhase(phase) {
  breathingText.textContent = phase.text;
  breathingCircle.style.transform = `scale(${phase.scale})`;

  circleProgress.style.transition = 'none';
  circleProgress.style.strokeDashoffset = `${circumference}`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      circleProgress.style.transition = `stroke-dashoffset ${phase.duration}ms linear`;
      circleProgress.style.strokeDashoffset = `0`;
    });
  });

  phaseTimeout = setTimeout(() => {
    phaseIndex = (phaseIndex + 1) % phases.length;
    if (isRunning) animatePhase(phases[phaseIndex]);
  }, phase.duration);
}

startButton.addEventListener('click', async () => {
  if (isRunning) {
    // Parar exercício e gravação
    isRunning = false;
    clearTimeout(phaseTimeout);
    breathingCircle.style.transform = 'scale(1)';
    circleProgress.style.transition = 'none';
    circleProgress.style.strokeDashoffset = `${circumference}`;
    breathingText.textContent = 'Pronto?';
    startButton.textContent = 'Iniciar Exercício';

    pararGravacao();
  } else {
    // Iniciar gravação primeiro, depois a animação
    try {
      await iniciarGravacao();
      isRunning = true;
      phaseIndex = 0;
      animatePhase(phases[phaseIndex]);
      startButton.textContent = 'Parar Exercício';
    } catch (err) {
      console.error('Erro ao acessar microfone:', err);
      alert('Erro ao iniciar gravação. Por favor, permita o acesso ao microfone.');
    }
  }
});

  // ---------- RESTO DO SEU CÓDIGO (charts, triggers, calendar) ----------

  function renderCharts(env_data) {
    const container = document.querySelector('.insights-container');
    env_data.labels = env_data.labels || [];

    Object.keys(env_data)
      .filter(key => key !== 'labels' && key !== 'triggers')
      .forEach(key => {
        const label = key
          .replace(/_/g, ' ')
          .replace(/\b\w/g, l => l.toUpperCase());

        if (charts[key]) {
          charts[key].data.labels = env_data.labels;
          charts[key].data.datasets[0].data = env_data[key];
          charts[key].update();
        } else {
          const card = document.createElement('div');
          card.className = 'card';

          const h3 = document.createElement('h3');
          h3.textContent = label;

          const canvas = document.createElement('canvas');
          canvas.id = key + 'Chart';

          card.append(h3, canvas);
          container.appendChild(card);

          const ctx = canvas.getContext('2d');
          charts[key] = new Chart(ctx, {
            type: 'line',
            data: {
              labels: env_data.labels,
              datasets: [{ label, data: env_data[key], fill: false }]
            },
            options: {
              responsive: true,
              scales: {
                x: { type: 'category', title: { display: true, text: 'Data Hora' } },
                y: { beginAtZero: false, title: { display: true, text: label } }
              },
              plugins: { datalabels: { display: false } }
            }
          });
        }
      });
  }

  function updateTriggers(triggers) {
    const triggersEl = document.getElementById('currentTriggers');
    triggersEl.innerHTML = '';
    const list = triggers.length ? triggers : ['Nenhum gatilho crítico no momento.'];
    list.forEach(txt => {
      const li = document.createElement('li');
      li.textContent = txt;
      triggersEl.appendChild(li);
    });
  }

  function loadData() {
    fetch('/api/data')
      .then(r => { if (!r.ok) throw new Error('Falha ao carregar /api/data'); return r.json(); })
      .then(({ env_data, usage_events }) => {
        renderCharts(env_data);
        updateTriggers(env_data.triggers);

        if (!calendar) {
          const calendarEl = document.getElementById('inhalerCalendar');
          calendar = new FullCalendar.Calendar(calendarEl, {
            initialView: 'dayGridMonth', locale: 'pt-br', height: 'auto', contentHeight: 350,
            events: usage_events,
            dateClick: handleDateClick,
            eventClick: handleEventClick
          });
          calendar.render();
        }
      })
      .catch(err => {
        console.error(err);
        alert('Erro ao carregar dados de insights. Veja o console.');
      });
  }

  function handleDateClick(info) {
    const date = info.dateStr;
    const existing = calendar.getEvents().filter(e => e.startStr === date);
    if (existing.length) {
      fetch(`/api/events?id=${existing[0].id}`, { method: 'DELETE' })
        .then(resp => { if (resp.ok) existing[0].remove(); });
    } else {
      fetch('/api/events', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ date })
      })
      .then(r => r.json())
      .then(evt => calendar.addEvent(evt));
    }
  }

  function handleEventClick(info) {
    fetch(`/api/events?id=${info.event.id}`, { method: 'DELETE' })
      .then(resp => { if (resp.ok) info.event.remove(); });
  }

  loadData();
  setInterval(loadData, 30000);

  // Toggle colapsar/expandir seção de gráficos
  const chartsSection = document.querySelector('.charts-section');
  const toggleBtn = document.querySelector('.toggle-charts-btn');
  const header = document.querySelector('.charts-header');

  function toggleCharts() {
    const isCollapsed = chartsSection.classList.toggle('collapsed');
    toggleBtn.setAttribute('aria-expanded', !isCollapsed);
  }
  toggleBtn.addEventListener('click', toggleCharts);
  header.addEventListener('click', toggleCharts);
});
