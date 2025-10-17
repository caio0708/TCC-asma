# -*- coding: utf-8 -*-
"""
MAX30102 Oximetria de Pulso - Leitor BLE (v7.4 - bleak)
------------------------------------------------------------------------------
- Versão completa adaptada para receber dados via Bluetooth Low Energy (BLE).
- Utiliza a biblioteca 'bleak' para se conectar ao ESP32-C3 ou similar.
- Hotkeys: [C] calibrar | [S] salvar | [L] carregar | [R] reset calib | [D] debug | [Q] sair.
"""

import os, sys, json, time, math, threading
from collections import deque
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner

try:
    from scipy.signal import butter, filtfilt, find_peaks, windows
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import msvcrt
    MSVCRT_AVAILABLE = True
except Exception:
    MSVCRT_AVAILABLE = False

# ==================== CONFIG BLE ====================
DEVICE_NAME = "SmartBand_ESP32-C3"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# ==================== CONFIG GERAL ====================
SAMPLE_RATE       = 100.0
BUFFER_SECONDS    = 6.0
BUFFER_SIZE       = int(SAMPLE_RATE * BUFFER_SECONDS)
SHORT_SECONDS     = 3.0
SHORT_SIZE        = int(SAMPLE_RATE * SHORT_SECONDS)
UPDATE_INTERVAL_S = 0.05

BPM_MIN, BPM_MAX  = 25, 250

WRIST_FILTER_LOW  = 0.6
WRIST_FILTER_HIGH = 4.0

# Qualidade / perdas
MIN_PI_PERCENT        = 0.2        # PI mínimo p/ aceitar leitura
MIN_AMP_IR            = 120.0      # amplitude mínima crua (ajuste se preciso)
SIGNAL_LOST_HOLD_S_BPM  = 1.5      # tempo de retenção após sinal ruim
SIGNAL_LOST_HOLD_S_SPO2 = 3.0

# Slew-rates (limites de rampa)
MAX_SPO2_SLEW_PER_S = 1.0
MAX_BPM_SLEW_PER_S  = 3.0

# Calibração
CALIB_FILE = "max30102_calibration.json"
SPO2_COEF_A_DEFAULT = 110.0
SPO2_COEF_B_DEFAULT = 25.0
SPO2_MANUAL_OFFSET  = 0.0
R_RANGE = (0.3, 2.2)

# Limites p/ regressões/escalas
HR_SCALE_LIM = (0.6, 1.4)
HR_BIAS_LIM  = (-20.0, 20.0)
SPO2_A_LIM   = (92.0, 114.0)
SPO2_B_LIM   = (8.0 , 40.0)
R_SCALE_LIM  = (0.30, 3.00)      # escala do R

DEBUG_MODE   = True
CLEAR_SCREEN = True

# ==================== ESTRUTURAS ====================
ir_buffer  = deque(maxlen=BUFFER_SIZE)
red_buffer = deque(maxlen=BUFFER_SIZE)

bpm_history  = deque(maxlen=8)
spo2_history = deque(maxlen=12)

KEYS = {"calibrate": False, "save": False, "load": False, "debug": False, "reset": False, "quit": False}

# ==================== CLASSES ====================
class Kalman1D:
    def __init__(self, x0=None, p0=25.0, q=0.15, r=4.0):
        self.x = x0; self.P = p0; self.q = q; self.r = r
    def predict(self): self.P = self.P + self.q
    def update(self, z, r_override=None):
        if z is None: return self.x
        R = r_override if r_override is not None else self.r
        K = self.P / (self.P + R)
        if self.x is None or math.isnan(self.x): self.x = z
        else: self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x
    def reset(self, x0=None, p0=25.0):
        self.x = x0; self.P = p0

class SlewLimiter:
    def __init__(self, max_per_sec, initial=None):
        self.max_per_sec = float(max_per_sec)
        self.last_val = initial; self.last_t = time.time()
    def apply(self, target):
        now = time.time(); dt = max(1e-3, now - self.last_t); self.last_t = now
        if target is None: return self.last_val
        if self.last_val is None: self.last_val = target; return target
        max_step = self.max_per_sec * dt; delta = float(target - self.last_val)
        if abs(delta) > max_step: self.last_val += math.copysign(max_step, delta)
        else: self.last_val = target
        return self.last_val
    def reset(self): self.last_val = None

class CalibrationStore:
    def __init__(self):
        self.spo2_A = SPO2_COEF_A_DEFAULT
        self.spo2_B = SPO2_COEF_B_DEFAULT
        self.spo2_offset = SPO2_MANUAL_OFFSET
        self.spo2_R_scale = 1.0
        self.hr_scale = 1.0; self.hr_bias = 0.0
        self.spo2_points = []
        self.hr_points   = []

    def load(self, path=CALIB_FILE):
        if not os.path.exists(path): return False
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        self.spo2_A = data.get("spo2_A", self.spo2_A)
        self.spo2_B = data.get("spo2_B", self.spo2_B)
        self.spo2_offset = data.get("spo2_offset", self.spo2_offset)
        self.spo2_R_scale = data.get("spo2_R_scale", self.spo2_R_scale)
        self.hr_scale = data.get("hr_scale", self.hr_scale)
        self.hr_bias  = data.get("hr_bias", self.hr_bias)
        self.spo2_points = data.get("spo2_points", [])
        self.hr_points   = data.get("hr_points", [])
        return True

    def save(self, path=CALIB_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "spo2_A": self.spo2_A, "spo2_B": self.spo2_B,
                "spo2_offset": self.spo2_offset, "spo2_R_scale": self.spo2_R_scale,
                "hr_scale": self.hr_scale, "hr_bias": self.hr_bias,
                "spo2_points": self.spo2_points, "hr_points": self.hr_points
            }, f, ensure_ascii=False, indent=2)
        return True

    @staticmethod
    def _linear_fit(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        X = np.vstack([x, np.ones_like(x)]).T
        sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        m, c = sol[0], sol[1]
        return float(m), float(c)

    def fit_spo2(self):
        if len(self.spo2_points) >= 2:
            R = np.array([p["R_mean"] for p in self.spo2_points], float)
            S = np.array([p["SpO2_ref"] for p in self.spo2_points], float)
            R_eff = self.spo2_R_scale * R
            m, c = self._linear_fit(-R_eff, S)
            self.spo2_B = float(np.clip(m, *SPO2_B_LIM))
            self.spo2_A = float(np.clip(c, *SPO2_A_LIM))
        elif len(self.spo2_points) == 1:
            p = self.spo2_points[0]
            Rm = float(p["R_mean"]); Sref = float(p["SpO2_ref"])
            num = (self.spo2_A + self.spo2_offset - Sref)
            den = (self.spo2_B * max(1e-6, Rm))
            self.spo2_R_scale = float(np.clip(num/den, *R_SCALE_LIM))
            S_calc = self.spo2_A - self.spo2_B * (self.spo2_R_scale * Rm) + self.spo2_offset
            self.spo2_offset += (Sref - S_calc)
            self.spo2_offset = float(np.clip(self.spo2_offset, -5.0, 5.0))

    def fit_hr(self):
        if len(self.hr_points) >= 2:
            x = np.array([p["bpm_meas"] for p in self.hr_points], float)
            y = np.array([p["bpm_ref"]  for p in self.hr_points], float)
            s, b = self._linear_fit(x, y)
            self.hr_scale = float(np.clip(s, *HR_SCALE_LIM))
            self.hr_bias  = float(np.clip(b, *HR_BIAS_LIM))
        elif len(self.hr_points) == 1:
            p = self.hr_points[0]
            self.hr_scale = 1.0
            self.hr_bias  = float(np.clip(p["bpm_ref"] - p["bpm_meas"], *HR_BIAS_LIM))

    def apply_spo2(self, R):
        R_eff = self.spo2_R_scale * float(R)
        return (self.spo2_A - self.spo2_B * R_eff) + self.spo2_offset

    def apply_bpm(self, bpm_meas):
        if bpm_meas is None: return None
        return (self.hr_scale * float(bpm_meas)) + self.hr_bias

CAL = CalibrationStore()

# ==================== SINAL ====================
def bandpass_filter(x, fs):
    x = np.asarray(x, float)
    if len(x) < 5: return x - np.median(x)
    if SCIPY_AVAILABLE:
        nyq = fs * 0.5
        low = max(0.01, WRIST_FILTER_LOW / nyq); high = min(0.99, WRIST_FILTER_HIGH / nyq)
        b, a = butter(3, [low, high], btype='band')
        return filtfilt(b, a, x)
    y = x - np.median(x)
    win = max(3, int(fs/10))
    c = np.convolve(y, np.ones(win)/win, mode='same')
    return y - c

def robust_ac_amplitude(x):
    if len(x) == 0: return 0.0
    return float(np.percentile(x, 90.0) - np.percentile(x, 10.0))

def perfusion_index(ac, dc):
    dc = max(1e-6, abs(dc)); return 100.0 * (float(ac) / dc)

def normalized_autocorr(x):
    x = np.asarray(x, float); x = x - np.mean(x)
    if np.allclose(np.std(x), 0.0): return np.zeros_like(x)
    ac = np.correlate(x, x, mode='full'); ac = ac[len(ac)//2:]; ac /= (ac[0] + 1e-9)
    return ac

def parabolic_refine(y, i):
    if i <= 0 or i >= len(y)-1: return float(i), float(y[i])
    y0, y1, y2 = y[i-1], y[i], y[i+1]; denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12: return float(i), float(y1)
    delta = 0.5 * (y0 - y2) / denom
    return float(i + delta), float(y1 - 0.25*(y0 - y2)*delta)

def bpm_from_autocorr(x, fs):
    ac = normalized_autocorr(x)
    min_lag = int(fs * 60.0 / BPM_MAX); max_lag = int(fs * 60.0 / BPM_MIN)
    min_lag = max(2, min_lag); max_lag = min(max_lag, len(ac)-2)
    if max_lag <= min_lag: return None, 0.0
    seg = ac[min_lag:max_lag]; i_rel = int(np.argmax(seg)); i = i_rel + min_lag
    i_ref, peak = parabolic_refine(ac, i)
    bpm = 60.0 / (i_ref / fs) if i_ref > 0 else None
    conf = max(0.0, min(1.0, float(peak)))
    return bpm, conf

def bpm_from_fft(x, fs):
    n = len(x)
    if n < 32: return None, 0.0
    x = np.asarray(x, float) - np.mean(x)
    if SCIPY_AVAILABLE: w = windows.hann(n, sym=False)
    else: w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/n)
    X = np.fft.rfft(x * w); P = np.abs(X)**2
    freqs = np.fft.rfftfreq(n, d=1.0/fs); bpm_axis = 60.0 * freqs
    mask = (bpm_axis >= BPM_MIN) & (bpm_axis <= BPM_MAX)
    if not np.any(mask): return None, 0.0
    idx = np.argmax(P[mask]); base = np.where(mask)[0][0]; i = base + idx
    i_ref, peak = parabolic_refine(P, i)
    df = (fs / n); dbpm = 60.0 * df
    bpm = float(bpm_axis[i]) + (i_ref - i) * dbpm
    conf = float(peak) / (np.sum(P[mask]) + 1e-9)
    return bpm, conf

def bpm_from_peaks(x, fs):
    x = np.asarray(x, float); xx = x - np.median(x)
    thr = np.percentile(xx, 75.0); min_dist = int(fs * 60.0 / BPM_MAX)
    if SCIPY_AVAILABLE: peaks, _ = find_peaks(xx, height=thr, distance=max(1, min_dist))
    else:
        peaks = []; i = 1; last_i = -10**9
        while i < len(xx)-1:
            if xx[i] > thr and xx[i] > xx[i-1] and xx[i] >= xx[i+1] and (i-last_i) >= min_dist:
                peaks.append(i); last_i = i
            i += 1
        peaks = np.array(peaks, dtype=int)
    if len(peaks) < 2: return None, 0.0
    ibis = np.diff(peaks) / fs
    ibis = ibis[(ibis > 60.0/BPM_MAX) & (ibis < 60.0/BPM_MIN)]
    if len(ibis) == 0: return None, 0.0
    bpm = 60.0 / np.median(ibis)
    mad = np.median(np.abs(ibis - np.median(ibis))) + 1e-9
    conf = float(1.0 / (1.0 + 10.0*mad)); conf = max(0.0, min(1.0, conf))
    return bpm, conf

def consensus_bpm(x_long, x_short, fs, prev_est=None):
    bpm_ac_l, c_ac_l = bpm_from_autocorr(x_long, fs)
    bpm_fft_l, c_fft_l = bpm_from_fft(x_long, fs)
    bpm_ibi_s, c_ibi_s = bpm_from_peaks(x_short, fs)
    candidates = []
    if bpm_ac_l: candidates.append((bpm_ac_l, c_ac_l*1.0))
    if bpm_fft_l: candidates.append((bpm_fft_l, c_fft_l*0.8))
    if bpm_ibi_s: candidates.append((bpm_ibi_s, c_ibi_s*1.2))
    if not candidates: return None, 0.0
    if prev_est is not None:
        adj = []
        for v, w in candidates:
            delta = abs(v - prev_est); prox = 1.0 / (1.0 + (delta/10.0)**2)
            adj.append((v, w * prox))
        candidates = adj
    weights = np.array([w for _, w in candidates], float) + 1e-9
    vals    = np.array([v for  v, _ in candidates], float)
    bpm = float(np.sum(vals * weights) / np.sum(weights))
    conf = float(np.clip(np.max(weights), 0.0, 1.0))
    if not (BPM_MIN <= bpm <= BPM_MAX): return None, 0.0
    return bpm, conf

# ==================== CÁLCULOS PRINCIPAIS ====================
def process_serial_line(line):
    try:
        if line.startswith('{') and line.endswith('}'):
            data = json.loads(line)
            if 'ir' in data and 'red' in data:
                ir_buffer.append(float(data['ir']))
                red_buffer.append(float(data['red']))
                return True
    except Exception: pass
    return False

def compute_r_ir_red(ir_raw, red_raw, fs):
    ir_raw = np.asarray(ir_raw, float); red_raw = np.asarray(red_raw, float)
    if len(ir_raw) < SHORT_SIZE: return None
    ir_f  = bandpass_filter(ir_raw, fs)
    red_f = bandpass_filter(red_raw, fs)
    ir_dc  = np.median(ir_raw); red_dc = np.median(red_raw)
    ir_ac  = robust_ac_amplitude(ir_f); red_ac = robust_ac_amplitude(red_f)
    if ir_dc <= 0 or red_dc <= 0 or ir_ac < 5.0: return None
    R = (red_ac / red_dc) / (ir_ac / ir_dc)
    PI = perfusion_index(ir_ac, ir_dc)
    ptp_raw = np.ptp(ir_raw[-SHORT_SIZE:])
    return {"ir_f": ir_f, "red_f": red_f, "ir_dc": ir_dc, "red_dc": red_dc,
            "ir_ac": ir_ac, "red_ac": red_ac, "R": R, "PI": PI, "PTP": ptp_raw}

def compute_spo2(comp):
    if comp is None: return None, None, 0.0, None, None
    R, PI = comp["R"], comp["PI"]
    if not (R_RANGE[0] <= R <= R_RANGE[1]): return None, R, 0.0, None, None
    if PI < MIN_PI_PERCENT: return None, R, 0.0, None, None
    R_eff = CAL.spo2_R_scale * R
    spo2_unclamped = CAL.spo2_A - CAL.spo2_B * R_eff + CAL.spo2_offset
    spo2 = float(np.clip(spo2_unclamped, 70.0, 100.0))
    r_center = 0.8
    conf_R = 1.0 / (1.0 + abs(R_eff - r_center))
    conf = float(np.clip(0.5*conf_R + 0.5*min(1.0, PI/1.5), 0.0, 1.0))
    return spo2, R, conf, R_eff, spo2_unclamped

def compute_bpm(ir_f_long, ir_f_short, fs, prev_bpm=None, comp=None):
    if comp is None or comp["PI"] < MIN_PI_PERCENT or comp["PTP"] < MIN_AMP_IR*0.3:
        return None, 0.0
    bpm, conf = consensus_bpm(ir_f_long, ir_f_short, fs, prev_est=prev_bpm)
    return bpm, conf

# ==================== CALLBACK DO BLE ====================
def notification_handler(sender, data):
    line = data.decode('utf-8', errors='ignore').strip()
    process_serial_line(line)

# ==================== HOTKEYS ====================
def keyboard_watcher():
    if not MSVCRT_AVAILABLE: return
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('c','C'): KEYS["calibrate"] = True
            elif ch in ('s','S'): KEYS["save"] = True
            elif ch in ('l','L'): KEYS["load"] = True
            elif ch in ('d','D'): KEYS["debug"] = True
            elif ch in ('r','R'): KEYS["reset"] = True
            elif ch in ('q','Q'): KEYS["quit"] = True; break
        time.sleep(0.03)

# ==================== CALIBRAÇÃO ====================
def perform_calibration(duration_s=15):
    t_end = time.time() + duration_s
    R_list = []; bpm_meas_list = []
    print(f"\n🔧 Calibração: fique em repouso por ~{duration_s} s e leia os valores no OXÍMETRO DE DEDO.")
    
    # Coleta de dados passiva, buffers são preenchidos em segundo plano
    while time.time() < t_end:
        time.sleep(0.1) 
    
    # Processa os dados acumulados nos buffers
    if len(ir_buffer) >= BUFFER_SIZE:
        ir_raw = list(ir_buffer); red_raw = list(red_buffer)
        comp = compute_r_ir_red(ir_raw, red_raw, SAMPLE_RATE)
        if comp is not None and (R_RANGE[0] <= comp["R"] <= R_RANGE[1]) and (comp["PI"] >= MIN_PI_PERCENT):
            R_list.append(comp["R"])
            ir_f_long  = comp["ir_f"][-BUFFER_SIZE:]; ir_f_short = comp["ir_f"][-SHORT_SIZE:]
            bpm_raw, c = compute_bpm(ir_f_long, ir_f_short, SAMPLE_RATE, prev_bpm=None, comp=comp)
            if bpm_raw is not None and c > 0.2: bpm_meas_list.append(bpm_raw)

    if not R_list:
        print("⚠️ Sinal insuficiente para calibrar (R/PI). Tente novamente.")
        return False

    try:
        ref_spo2 = float(input("Digite a SpO₂ de REFERÊNCIA (dedo) em %: ").strip().replace(',', '.'))
        ref_bpm  = float(input("Digite o BPM de REFERÊNCIA (dedo): ").strip().replace(',', '.'))
    except Exception:
        print("⚠️ Entrada inválida. Calibração abortada.")
        return False

    R_mean = float(np.median(R_list))
    CAL.spo2_points.append({"R_mean": R_mean, "SpO2_ref": ref_spo2})
    CAL.fit_spo2()

    if bpm_meas_list:
        bpm_meas = float(np.median(bpm_meas_list))
        CAL.hr_points.append({"bpm_meas": bpm_meas, "bpm_ref": ref_bpm})
        CAL.fit_hr()

    print("✅ Calibração aplicada.")
    print("   SpO₂: A={:.3f}, B={:.3f}, R_scale={:.3f}, offset={:+.2f} | pts={}".format(
        CAL.spo2_A, CAL.spo2_B, CAL.spo2_R_scale, CAL.spo2_offset, len(CAL.spo2_points)))
    print("   BPM:  scale={:.4f}, bias={:+.2f} | pts={}".format(
        CAL.hr_scale, CAL.hr_bias, len(CAL.hr_points)))
    return True

# ==================== UI ====================
def display_panel(bpm_disp, spo2_disp, r_val, r_eff, spo2_unc, pi_val, conf_bpm, conf_spo2, signal_ok):
    if CLEAR_SCREEN: os.system('cls' if os.name == 'nt' else 'clear')
    print("═"*84)
    print("   MONITOR CARDÍACO & OXIMETRIA (MAX30102) - v7.4 BLE Reader")
    print("═"*84)
    print(f"⏰ {time.strftime('%H:%M:%S')} | Amostras: {len(ir_buffer)}/{BUFFER_SIZE} | Fs={SAMPLE_RATE:.1f} Hz")
    print("-"*84)
    s_bpm  = f"{bpm_disp:5.0f} bpm" if bpm_disp is not None else ("Sem sinal" if not signal_ok else "Buscando...")
    s_spo2 = f"{spo2_disp:5.1f} %" if spo2_disp is not None else ("Sem sinal" if not signal_ok else "Calculando...")
    print("📊 SINAIS VITAIS (após suavização):")
    print(f"   ❤️ BPM:   {s_bpm}   (conf={conf_bpm*100:4.0f}%)")
    print(f"   🟢 SpO₂:  {s_spo2}   (conf={conf_spo2*100:4.0f}%)")
    print("-"*84)
    print("⚙️  Calibração atual:")
    print("   SpO₂ = A − B·(R_scale·R) + offset  |  A={:.2f}  B={:.2f}  R_scale={:.3f}  offset={:+.2f} | pts={}".format(
        CAL.spo2_A, CAL.spo2_B, CAL.spo2_R_scale, CAL.spo2_offset, len(CAL.spo2_points)))
    print("   BPM_cal = scale·BPM_meas + bias   |  scale={:.4f}  bias={:+.2f} | pts={}".format(
        CAL.hr_scale, CAL.hr_bias, len(CAL.hr_points)))
    if DEBUG_MODE:
        rv   = f"{(r_val if r_val is not None else float('nan')):.4f}" if r_val is not None else "---"
        reff = f"{(r_eff if r_eff is not None else float('nan')):.4f}" if r_eff is not None else "---"
        piv  = f"{pi_val:.2f} %" if pi_val is not None else "---"
        sunc = f"{spo2_unc:5.1f} %" if spo2_unc is not None else "---"
        print("-"*84)
        print(f"🛠️ DEBUG: R={rv} | R_eff={reff} | SpO2_calc={sunc} | PI={piv} | keys: [C]alib [S]alvar [L]carregar [R]reset [D]debug [Q]sair")
    print("═"*84)

# ==================== MAIN ====================
async def main():
    print("Iniciando... Procurando por dispositivo BLE:", DEVICE_NAME)
    
    device = None
    while device is None:
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            for d in devices:
                if d.name and d.name.lower() == DEVICE_NAME.lower():
                    device = d
                    print(f"✅ Dispositivo encontrado: {device.name} [{device.address}]")
                    break
            if not device:
                print(f"Dispositivo '{DEVICE_NAME}' não encontrado. Tentando novamente em 5s...")
        except Exception as e:
            print(f"❌ ERRO durante a busca: {e}. Tentando novamente...")
            await asyncio.sleep(5)
    
    if CAL.load(): print("ℹ️  Calibração carregada de", CALIB_FILE)
    else:          print("ℹ️  Sem arquivo de calibração. Usando coeficientes padrão.")

    if MSVCRT_AVAILABLE:
        threading.Thread(target=keyboard_watcher, daemon=True).start()

    kalman_bpm  = Kalman1D(x0=None, p0=25.0, q=0.25, r=6.0)
    kalman_spo2 = Kalman1D(x0=None, p0=16.0, q=0.10, r=3.0)
    slew_bpm    = SlewLimiter(MAX_BPM_SLEW_PER_S)
    slew_spo2   = SlewLimiter(MAX_SPO2_SLEW_PER_S)
    
    prev_bpm_for_consensus = None
    last_display_t = 0.0
    last_valid_bpm_t = 0.0
    last_valid_spo2_t = 0.0

    def reset_bpm_pipeline():
        kalman_bpm.reset(); slew_bpm.reset(); bpm_history.clear()

    def reset_spo2_pipeline():
        kalman_spo2.reset(); slew_spo2.reset(); spo2_history.clear()

    print("Conectando ao dispositivo...")
    async with BleakClient(device) as client:
        if not client.is_connected:
            print("❌ Falha ao conectar.")
            return

        print(f"✅ Conectado a {DEVICE_NAME}")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("✅ Assinatura de notificações ativa. Recebendo dados...")

        try:
            while True:
                if KEYS["quit"]: break
                if KEYS["debug"]:
                    global DEBUG_MODE
                    DEBUG_MODE = not DEBUG_MODE; KEYS["debug"] = False
                if KEYS["load"]:
                    print("✅ Calibração carregada." if CAL.load() else "⚠️ Falha ao carregar calibração.")
                    KEYS["load"] = False
                if KEYS["save"]:
                    print("💾 Calibração salva em", CALIB_FILE) if CAL.save() else print("⚠️ Falha ao salvar calibração.")
                    KEYS["save"] = False
                if KEYS["reset"]:
                    CAL.spo2_A = SPO2_COEF_A_DEFAULT
                    CAL.spo2_B = SPO2_COEF_B_DEFAULT
                    CAL.spo2_offset = 0.0
                    CAL.spo2_R_scale = 1.0
                    CAL.hr_scale = 1.0
                    CAL.hr_bias  = 0.0
                    CAL.spo2_points.clear()
                    CAL.hr_points.clear()
                    print("♻️  Calibração resetada (A/B/offset/R_scale e pontos).")
                    KEYS["reset"] = False
                if KEYS["calibrate"]:
                    perform_calibration(); KEYS["calibrate"] = False

                bpm_disp = None; spo2_disp = None; r_val = None; r_eff = None; spo2_unc = None
                pi_val = None; conf_bpm = 0.0; conf_spo2 = 0.0; signal_ok = False
                now = time.time()

                if len(ir_buffer) >= BUFFER_SIZE:
                    ir_raw  = list(ir_buffer)[-BUFFER_SIZE:]
                    red_raw = list(red_buffer)[-BUFFER_SIZE:]
                    comp = compute_r_ir_red(ir_raw, red_raw, SAMPLE_RATE)

                    if comp is not None:
                        pi_val = comp["PI"]
                        signal_ok = (comp["PI"] >= MIN_PI_PERCENT) and (comp["PTP"] >= MIN_AMP_IR*0.3)

                    spo2_meas, r_val, conf_spo2, r_eff, spo2_unc = compute_spo2(comp)
                    if spo2_meas is not None and signal_ok:
                        last_valid_spo2_t = now
                        kalman_spo2.predict()
                        r_meas_spo2 = 2.0 + (1.0 - min(1.0, conf_spo2)) * 8.0
                        spo2_k = kalman_spo2.update(spo2_meas, r_override=r_meas_spo2)
                        spo2_disp = slew_spo2.apply(spo2_k)
                        if spo2_disp is not None: spo2_history.append(spo2_disp)
                    else:
                        if (now - last_valid_spo2_t) > SIGNAL_LOST_HOLD_S_SPO2:
                            reset_spo2_pipeline()
                            spo2_disp = None

                    bpm_meas = None
                    if comp is not None:
                        ir_f_long  = comp["ir_f"][-BUFFER_SIZE:]
                        ir_f_short = comp["ir_f"][-SHORT_SIZE:]
                        bpm_meas, conf_bpm = compute_bpm(ir_f_long, ir_f_short, SAMPLE_RATE, prev_bpm=prev_bpm_for_consensus, comp=comp)
                        prev_bpm_for_consensus = bpm_meas if bpm_meas is not None else prev_bpm_for_consensus

                    if bpm_meas is not None and signal_ok:
                        bpm_corr = CAL.apply_bpm(bpm_meas)
                        if abs(CAL.hr_scale) < 0.2: bpm_corr = bpm_meas + CAL.hr_bias
                        last_valid_bpm_t = now
                        kalman_bpm.predict()
                        r_meas_bpm = 4.0 + (1.0 - min(1.0, conf_bpm)) * 16.0
                        bpm_k = kalman_bpm.update(bpm_corr, r_override=r_meas_bpm)
                        bpm_disp = slew_bpm.apply(bpm_k)
                        if bpm_disp is not None: bpm_history.append(bpm_disp)
                    else:
                        if (now - last_valid_bpm_t) > SIGNAL_LOST_HOLD_S_BPM:
                            reset_bpm_pipeline()
                            bpm_disp = None
                
                if time.time() - last_display_t >= UPDATE_INTERVAL_S:
                    disp_bpm  = (np.median(bpm_history)  if bpm_history  else None)
                    disp_spo2 = (np.median(spo2_history) if spo2_history else None)
                    display_panel(disp_bpm, disp_spo2, r_val, r_eff, spo2_unc, pi_val, conf_bpm, conf_spo2, signal_ok)
                    last_display_t = time.time()

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⏹️ Programa encerrado pelo usuário.")
        except Exception as e:
            print(f"\n❌ ERRO INESPERADO: {e}")
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)
            print("Notificações paradas.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Erro ao iniciar o programa: {e}")