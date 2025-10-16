# -*- coding: utf-8 -*-
"""
MAX30102 MQTT Reader — Robust HR (BPM) + SpO2 (v2)
--------------------------------------------------
Correções principais (em relação à versão anterior):
1) SpO2 não aparecia: agora o DC é estimado por "baseline" (média móvel ~1,2 s)
   nos sinais **brutos** IR/RED e o AC é medido no pico do IR filtrado.
   → Garante AC/DC válidos por batimento; se ainda faltar batimento, usa fallback RMS.
2) BPM alto (p.ex., ~100 quando deveria ~70): reforçada a detecção de picos
   (distance mínimo 0,45 s) e adicionada **estimativa por autocorrelação**.
   → Se HR por picos divergir muito do HR por ACF, a ACF prevalece.
3) Janela de 8 s (800 amostras @100 Hz) para resposta mais rápida.
4) Logs opcionais de depuração (quantidade de picos e batimentos válidos).
ATENÇÃO: Uso educacional. Não utilizar para decisões clínicas.
"""

import os
import json
import time
import random
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

TOPIC_OX_JSON  = "sensorestcc/max30102"       # JSON: {"seq":N,"t_us":micros(),"ir":<u32>,"red":<u32>}
TOPIC_IR_COMPAT  = "sensorestcc/max30102-ir"  # Legado: IR sozinho
TOPIC_RED_COMPAT = "sensorestcc/max30102-red" # Legado: RED sozinho
TOPIC_TEMP = "sensorestcc/temperatura-ambiente"
TOPIC_HUM  = "sensorestcc/umidade"

# Janela de processamento (8 s @ 100 Hz)
SAMPLES_TO_PROCESS   = 800
SAMPLE_RATE_HINT_HZ  = 100.0
DEBUG_PRINTS         = False   # True para ver contagens de picos/batimentos

CLEAR_EVERY_UPDATE = True

# =============================================================================
# BUFFERS
# =============================================================================
ir_buffer = deque(maxlen=SAMPLES_TO_PROCESS)
red_buffer = deque(maxlen=SAMPLES_TO_PROCESS)
t_buffer  = deque(maxlen=SAMPLES_TO_PROCESS)  # segundos (monotônico)

last_temperature = None
last_humidity    = None

print("========================================================================")
print("AVISO: Script educacional. Não utilizar para decisões clínicas.")
print("========================================================================")
time.sleep(1)

# =============================================================================
# FUNÇÕES DE SINAL
# =============================================================================
def _moving_average(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def _bandpass_np(x, fs):
    # Fallback leve ao SciPy: HP ~1.0 s + LP ~0.1 s ≈ 0,5–5 Hz
    x_hp = x - _moving_average(x, int(max(3, fs*1.0)))
    return _moving_average(x_hp, int(max(3, fs*0.1)))

def _baseline_ma(x, fs, win_sec=1.2):
    w = max(3, int(round(fs*win_sec)))
    return _moving_average(x, w)

def _compute_fs(timestamps_s, n, fs_hint):
    if timestamps_s is None or len(timestamps_s) < 2:
        return float(fs_hint)
    dt = float(timestamps_s[-1]) - float(timestamps_s[0])
    return float(n-1)/dt if dt > 0 else float(fs_hint)

def _find_peaks_numpy(sig, fs):
    peaks = []
    min_dist = int(max(1, 0.45*fs))  # >= ~133 bpm -> evita entalhe dicrótico
    thr = np.mean(sig) + max(1e-9, 0.5*np.std(sig))
    i = 1
    last = -10**9
    n = len(sig)
    while i < n-1:
        if sig[i] > thr and sig[i] > sig[i-1] and sig[i] > sig[i+1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
                i += min_dist
                continue
        i += 1
    return np.asarray(peaks, int)

def _hr_from_acf(sig, fs, min_bpm=40, max_bpm=180):
    x = np.asarray(sig, float)
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    # Procura no intervalo de lags permitido
    min_lag = int(fs*60.0/max_bpm)  # 0.33 s p/ 180 bpm
    max_lag = int(fs*60.0/min_bpm)  # 1.5 s p/ 40 bpm
    if max_lag <= min_lag+1 or max_lag >= len(ac):
        return None
    seg = ac[min_lag:max_lag]
    lag = int(min_lag + np.argmax(seg))
    if lag <= 0:
        return None
    return 60.0 * fs / lag

def calculate_vitals(ir_data, red_data, timestamps_s=None, fs_hint=SAMPLE_RATE_HINT_HZ):
    """
    Retorna (spo2, heart_rate, hrv_sdnn_ms, sqi) ou (None, None, None, None).
    """
    ir  = np.asarray(ir_data, dtype=float)
    red = np.asarray(red_data, dtype=float)
    n = len(ir)
    if n < 200 or n != len(red):
        return None, None, None, None

    fs = _compute_fs(timestamps_s, n, fs_hint)

    # --- Filtragem ---
    try:
        from scipy.signal import butter, filtfilt, find_peaks
        b, a = butter(2, [0.5/(fs/2), 5.0/(fs/2)], btype='band')
        ir_f  = filtfilt(b, a, ir)
        red_f = filtfilt(b, a, red)
        use_scipy = True
    except Exception:
        ir_f  = _bandpass_np(ir, fs)
        red_f = _bandpass_np(red, fs)
        use_scipy = False

    # --- Detecção de picos no IR filtrado ---
    min_dist = int(max(1, 0.45*fs))  # >= ~133 bpm
    prom = 0.5 * np.nanstd(ir_f)
    if use_scipy:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ir_f,
                              distance=min_dist,
                              prominence=max(1e-6, prom),
                              width=(int(0.06*fs), int(0.50*fs)))
    else:
        peaks = _find_peaks_numpy(ir_f, fs)

    peaks = np.asarray(peaks, int)
    if len(peaks) < 2:
        return None, None, None, None

    # --- RR → BPM (por picos) ---
    rr_s = np.diff(peaks) / fs
    rr_s = rr_s[(rr_s > 0.35) & (rr_s < 2.0)]  # 30–171 bpm
    heart_rate_pk = 60.0 / np.median(rr_s) if len(rr_s) >= 3 else None
    hrv_sdnn = np.std(rr_s * 1000.0) if len(rr_s) >= 15 else None

    # --- HR por autocorrelação (robusto a picos falsos) ---
    heart_rate_acf = _hr_from_acf(ir_f, fs, 40, 180)

    # Combine: se os dois existem e divergem > 15 bpm, confia na ACF
    if heart_rate_acf is not None and heart_rate_pk is not None:
        if abs(heart_rate_acf - heart_rate_pk) > 15:
            heart_rate = heart_rate_acf
        else:
            heart_rate = float(np.median([heart_rate_acf, heart_rate_pk]))
    else:
        heart_rate = heart_rate_acf if heart_rate_acf is not None else heart_rate_pk

    # --- SpO2 por batimento (AC/DC) ---
    # DC: baseline por média móvel nos sinais brutos (não filtrados)
    ir_dc  = _baseline_ma(ir,  fs, win_sec=1.2)
    red_dc = _baseline_ma(red, fs, win_sec=1.2)
    eps = 1e-9

    Rs = []
    for pk in peaks:
        # AC no pico = pico do filtrado - baseline (DC) no tempo do pico
        ac_ir  = float(ir_f[pk]  - (ir_dc[pk]  - np.mean(ir)))  # corrige offset
        ac_red = float(red_f[pk] - (red_dc[pk] - np.mean(red)))
        dc_ir  = float(max(eps, ir_dc[pk]))
        dc_red = float(max(eps, red_dc[pk]))

        ac_ir_abs  = abs(ac_ir)
        ac_red_abs = abs(ac_red)
        if ac_ir_abs <= eps or ac_red_abs <= eps:
            continue

        r = ( (ac_red_abs/dc_red) / (ac_ir_abs/dc_ir) )
        if 0.1 < r < 4.0:
            Rs.append(r)

    spo2 = None
    if len(Rs) >= 3:
        R_med = float(np.median(Rs))
        spo2 = max(0.0, min(100.0, 104.0 - 17.0 * R_med))
    else:
        # Fallback RMS por janela inteira (menos preciso, mas evita "Calculando...")
        # AC: desvio padrão do filtrado; DC: média do bruto
        ac_ir_rms  = float(np.std(ir_f))
        ac_red_rms = float(np.std(red_f))
        dc_ir_mean  = float(max(eps, np.mean(ir_dc)))
        dc_red_mean = float(max(eps, np.mean(red_dc)))
        if ac_ir_rms > eps and dc_ir_mean > eps and dc_red_mean > eps:
            R_win = (ac_red_rms/dc_red_mean) / (ac_ir_rms/dc_ir_mean)
            if 0.1 < R_win < 4.0:
                spo2 = max(0.0, min(100.0, 104.0 - 17.0 * R_win))

    # --- SQI simples ---
    window_s = n / fs
    exp_min_beats = window_s * 35.0 / 60.0  # mínimo esperado p/ 35 bpm
    sqi = None
    if exp_min_beats > 0:
        sqi = max(0.0, min(1.0, (len(rr_s)) / max(1.0, exp_min_beats)))

    if DEBUG_PRINTS:
        print(f"[DEBUG] fs={fs:.1f}Hz, peaks={len(peaks)}, RR_valid={len(rr_s)}, Rbeats={len(Rs)}")

    return spo2, heart_rate, hrv_sdnn, sqi

# =============================================================================
# INTERPRETAÇÃO (opcional)
# =============================================================================
def interpret_for_asthma(spo2, hr, hrv, temp, hum):
    status = "Estável"
    warnings = []

    if spo2 is not None:
        if spo2 < 90:
            status = "ALERTA CRÍTICO"
            warnings.append(f"SpO₂ muito baixo ({spo2:.1f}%). Procure ajuda médica.")
        elif spo2 < 94:
            status = "Atenção"
            warnings.append(f"SpO₂ baixo ({spo2:.1f}%). Monitorar de perto.")

    if hr is not None:
        if hr > 120:
            status = "Alerta"
            warnings.append(f"Taquicardia severa ({hr:.0f} bpm).")
        elif hr > 100:
            warnings.append(f"Frequência cardíaca elevada ({hr:.0f} bpm).")

    if hrv is not None and hrv < 20:
        warnings.append(f"VFC baixa ({hrv:.1f} ms).")

    if temp is not None and temp < 15:
        warnings.append(f"Ambiente frio ({temp:.1f} °C).")
    if hum is not None and hum > 70:
        warnings.append(f"Ambiente muito úmido ({hum:.1f}%).")
    if hum is not None and hum < 30:
        warnings.append(f"Ambiente muito seco ({hum:.1f}%).")

    if not warnings:
        warnings.append("Sinais e ambiente dentro dos limites usuais.")

    return status, warnings

# =============================================================================
# MQTT
# =============================================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado ao Broker MQTT.")
        client.subscribe(TOPIC_OX_JSON)
        client.subscribe(TOPIC_IR_COMPAT)
        client.subscribe(TOPIC_RED_COMPAT)
        client.subscribe(TOPIC_TEMP)
        client.subscribe(TOPIC_HUM)
    else:
        print(f"Falha na conexão MQTT. Código {rc}")

def on_message(client, userdata, msg):
    global last_temperature, last_humidity

    try:
        payload_str = msg.payload.decode().strip()
    except Exception:
        return

    if msg.topic == TOPIC_OX_JSON:
        try:
            obj = json.loads(payload_str)
            ir_val  = float(obj.get("ir"))
            red_val = float(obj.get("red"))
            t_sec = float(obj.get("t_us", 0.0)) / 1e6 if ("t_us" in obj) else time.time()
            ir_buffer.append(ir_val)
            red_buffer.append(red_val)
            t_buffer.append(t_sec)
        except Exception:
            pass

    elif msg.topic == TOPIC_IR_COMPAT:
        try:
            val = float(payload_str)
            ir_buffer.append(val)
            t_buffer.append(time.time())  # timestamp menos preciso
        except Exception:
            pass
    elif msg.topic == TOPIC_RED_COMPAT:
        try:
            val = float(payload_str)
            red_buffer.append(val)
        except Exception:
            pass

    elif msg.topic == TOPIC_TEMP:
        try:
            last_temperature = float(payload_str)
        except Exception:
            pass
    elif msg.topic == TOPIC_HUM:
        try:
            last_humidity = float(payload_str)
        except Exception:
            pass

    check_and_process_data()

def check_and_process_data():
    n = min(len(ir_buffer), len(red_buffer))
    if n >= SAMPLES_TO_PROCESS:
        ir = np.asarray(list(ir_buffer)[-SAMPLES_TO_PROCESS:], dtype=float)
        red = np.asarray(list(red_buffer)[-SAMPLES_TO_PROCESS:], dtype=float)
        t = None
        if len(t_buffer) == len(ir_buffer):
            t = np.asarray(list(t_buffer)[-SAMPLES_TO_PROCESS:], dtype=float)

        ir_buffer.clear(); red_buffer.clear(); t_buffer.clear()

        spo2, hr, hrv, sqi = calculate_vitals(ir, red, t, fs_hint=SAMPLE_RATE_HINT_HZ)
        status, warnings = interpret_for_asthma(spo2, hr, hrv, last_temperature, last_humidity)

        if CLEAR_EVERY_UPDATE:
            os.system('cls' if os.name == 'nt' else 'clear')

        print("--- PAINEL DE ANÁLISE ---")
        print(f"Atualizado em: {time.strftime('%H:%M:%S')}")
        print("-------------------------")
        print(f"SpO₂:     {spo2:.1f} %" if spo2 is not None else "SpO₂:     Calculando...")
        print(f"BPM:      {hr:.0f} bpm" if hr is not None else "BPM:      Calculando...")
        print(f"VFC SDNN: {hrv:.1f} ms" if hrv is not None else "VFC SDNN: Calculando...")
        if sqi is not None:
            print(f"Qualidade do sinal (SQI): {int(sqi*100)}%")
        print("------ Ambiente ------")
        print(f"Temperatura: {last_temperature:.1f} °C" if last_temperature is not None else "Temp: Aguardando...")
        print(f"Umidade:     {last_humidity:.1f} %" if last_humidity is not None else "Umid: Aguardando...")
        print("------ Análise -------")
        for w in warnings:
            print(f"- {w}")
        print("-------------------------\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    random_id = random.randint(1000, 9999)
    client = mqtt.Client(client_id=f"max30102_rx_{random_id}")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print("Conectando ao broker MQTT...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nEncerrando...")
        try:
            client.disconnect()
        except Exception:
            pass
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
