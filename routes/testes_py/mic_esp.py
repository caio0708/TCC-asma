#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import numpy as np
import sys
import os
import pickle
import time
from scipy.signal import butter,filtfilt, decimate
from scipy import signal
from scipy.integrate import simpson
import librosa

# --- ETAPA 2: CONFIGURAÇÕES GERAIS E MQTT ---
MQTT_BROKER = "broker.hivemq.com" 
MQTT_TOPIC = "audio/stream/tosse-paciente-1"

# Parâmetros do áudio (DEVEM SER IGUAIS AOS DO ESP32 E DO MODELO)
# Os modelos deste projeto foram treinados com 16000Hz.
SAMPLE_RATE = 16000
SAMPLE_TYPE = np.int16

# Buffer para acumular os dados de áudio recebidos via MQTT
audio_buffer = bytearray()

def preprocess_cough(x, fs, cutoff=6000, filter_=True, downsample=True):
    """
    CORRIGIDO E MELHORADO: Processa o sinal removendo DC offset,
    convertendo para float, filtrando e fazendo downsampling.
    """
    # Etapa 1: Converter para float para cálculos
    x_float = x.astype(np.float32)

    # Etapa 2: Remover o DC Offset para centralizar o sinal em 0
    x_float = x_float - np.mean(x_float)

    # Etapa 3: Normalizar pelo valor máximo de int16 para preservar a amplitude
    x_normalized = x_float / 32768.0

    # O resto do processamento usa o sinal limpo e normalizado
    if len(x_normalized.shape) > 1:
        x_normalized = np.mean(x_normalized, axis=1)

    fs_downsample = cutoff * 2
    
    if filter_:
        b, a = butter(4, fs_downsample / fs, btype='lowpass')
        x_normalized = filtfilt(b, a, x_normalized)

    if downsample:
        x_normalized = decimate(x_normalized, int(fs / fs_downsample))
    
    fs_new = fs_downsample
    return x_normalized, fs_new

def classify_cough(x, fs, model, scaler):
    """Classify whether an inputted signal is a cough or not using filtering, feature extraction, and ML classification
    Inputs: 
        x: (float array) raw cough signal
        fs: (int) sampling rate of raw signal
        model: cough classification ML model loaded from file
    Outputs:
        result: (float) probability that a given file is a cough 
    """
    try: 
        x,fs = preprocess_cough(x,fs)
        data = (fs,x)
        FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
        features_fct_list = ['EEPD','ZCR','RMSP','DF','spectral_features','SF_SSTD','SSL_SD','MFCC','CF','LGTH','PSD']
        feature_values_vec = []
        obj = features(FREQ_CUTS)
        for feature in features_fct_list:
            feature_values, feature_names = getattr(obj,feature)(data)
            for value  in feature_values:
                if isinstance(value,np.ndarray):
                    feature_values_vec.append(value[0])
                else:
                    feature_values_vec.append(value)
        feature_values_scaled = scaler.transform(np.array(feature_values_vec).reshape(1,-1))
        result = model.predict_proba(feature_values_scaled)[:,1]
        return result[0]
    except:
        "Feature extraction fails when the audio is completely silent"
        return 0

# --- ETAPA 3: CARREGAR OS MODELOS DE MACHINE LEARNING ---
def carregar_modelos_tosse():
    """Carrega o classificador de tosse e o normalizador (scaler) do disco."""
    try:
        model_path = os.path.join('model_artifacts', 'cough_classifier')
        scaler_path = os.path.join('model_artifacts', 'cough_classification_scaler')
        
        print(f"Carregando modelo de: {model_path}")
        cough_model = pickle.load(open(model_path, 'rb'))
        
        print(f"Carregando normalizador de: {scaler_path}")
        cough_scaler = pickle.load(open(scaler_path, 'rb'))
        
        print("Modelo e normalizador carregados com sucesso.")
        return cough_model, cough_scaler
        
    except FileNotFoundError:
        print("ERRO: Arquivos de modelo não encontrados na pasta 'model_artifacts'.")
        print("Verifique se as pastas 'model_artifacts' estão no lugar correto.")
        return None, None

# --- ETAPA 4: FUNÇÃO DE DETECÇÃO REAL ---
def analisar_segmento_para_tosse(audio_segment, model, scaler):
    """
    Analisa um segmento de áudio usando o modelo carregado.
    
    Args:
        audio_segment (np.array): Array numpy com amostras de áudio (int16).
        model: O modelo classificador carregado.
        scaler: O normalizador (scaler) carregado.
    """
    # A função `classify_cough` faz todo o trabalho pesado:
    # extrai as features, normaliza e faz a predição.
    probabilidade = classify_cough(audio_segment, SAMPLE_RATE, model, scaler)
    
    # Imprime o resultado da análise
    print(f"Segmento analisado. Probabilidade de tosse: {probabilidade:.4f}")
    
    # Lógica de decisão: se a probabilidade for alta, sinalize!
    if probabilidade > 0.8:  # Você pode ajustar este limiar
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!  TOSSE DETECTADA (Confiança: {probabilidade*100:.1f}%) !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

# --- ETAPA 5: LÓGICA DO CLIENTE MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback para quando o cliente se conecta ao broker."""
    if rc == 0:
        print(f"Conectado ao Broker MQTT! Inscrevendo-se no tópico '{MQTT_TOPIC}'")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Falha na conexão, código de retorno {rc}\n")

def on_message(client, userdata, msg):
    """
    Callback para quando uma mensagem de áudio chega.
    Acumula os dados e dispara a análise quando um segmento completo é formado.
    """
    # Os modelos de ML funcionam melhor com segmentos de tamanho fixo.
    # Vamos processar a cada 1 segundo de áudio.
    global audio_buffer
    
    # userdata é onde armazenamos o modelo e o scaler
    model, scaler = userdata 

    audio_buffer.extend(msg.payload)
    
    bytes_por_segmento = SAMPLE_RATE * SAMPLE_TYPE().itemsize # 1 segundo de áudio
    
    while len(audio_buffer) >= bytes_por_segmento:
        segmento_bytes = audio_buffer[:bytes_por_segmento]
        del audio_buffer[:bytes_por_segmento]

        # Converte os bytes em um array numpy, que é o formato que a função de análise espera
        audio_samples = np.frombuffer(segmento_bytes, dtype=SAMPLE_TYPE)
        
        # Chama a função de análise real
        analisar_segmento_para_tosse(audio_samples, model, scaler)

def main():
    """Função principal que inicializa tudo."""
    # Carrega os modelos primeiro. Se falhar, o script para.
    tosse_model, tosse_scaler = carregar_modelos_tosse()
    if tosse_model is None or tosse_scaler is None:
        sys.exit(1)

    # Configura o cliente MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Armazena os modelos carregados no 'userdata' do cliente.
    # Assim, eles ficam acessíveis dentro da função 'on_message'.
    client.user_data_set((tosse_model, tosse_scaler))

    try:
        print(f"Conectando ao broker MQTT em {MQTT_BROKER}...")
        client.connect(MQTT_BROKER, 1883, 60)
    except Exception as e:
        print(f"Não foi possível conectar ao broker MQTT: {e}")
        sys.exit(1)

    client.loop_forever()











class features:
    # output should be  a np.array 
    # names should be a list of the size of output
    # add the number of features in output of each function
    n_std_dev = 1
    n_dummy = 2
    n_EEPD = 19
    n_PRE = 1
    n_ZCR = 1
    n_RMSP = 1
    n_DF = 1
    n_spectral_features = 6
    n_SF_SSTD = 2
    n_MFCC = 26
    n_CF = 1
    n_LGTH = 1
    n_SSL_SD = 2
    
    def __init__(self, FREQ_CUTS):
        self.FREQ_CUTS = FREQ_CUTS # list of Frequency Bands for the PSD
        self.n_PSD = len(FREQ_CUTS)
        
    def std_dev(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['std_dev'] # list of output features  
        std_deviation = np.ones((1,1))*np.std(data[1])
        return std_deviation, names
    
    def dummy(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature (MUST BE AN ARRAY)
        names = ['dummy_feature_2','dummy_3']
        return np.array([1.,2.]), names
    
    def fft(self,data):
        """
        Compute the spectrum using FFT
        """
        fs, cough = data
        fftdata = np.fft.rfft(cough)
        return fftdata
    
    # Envelope Energy Peak Detection
    def EEPD(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = []
        fs,cough = data
        fNyq = fs/2
        nPeaks = []
        freq_step = 50
        for fcl in range(50,1000,freq_step):
            names = names + ['EEPD'+str(fcl)+'_'+str(fcl+freq_step)]
            fc = [fcl/fNyq, (fcl+50)/fNyq]
            b, a = butter(1, fc, btype='bandpass')
            bpFilt = filtfilt(b, a, cough)
            b,a = butter(2, 10/fNyq, btype='lowpass')
            eed = filtfilt(b, a, bpFilt**2)
            eed = eed/np.max(eed+1e-17)
            peaks,_ = signal.find_peaks(eed)
            nPeaks.append(peaks.shape[0])
        return np.array(nPeaks), names

    # Phase Power Ratio Estimation
    def PRE(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Power_Ratio_Est']
        fs,cough = data
        phaseLen = int(cough.shape[0]//3)
        P1 = cough[:phaseLen]
        P2 = cough[phaseLen:2*phaseLen]
        P3 = cough[2*phaseLen:]
        f = np.fft.fftfreq(phaseLen, 1/fs)
        P1 = np.abs(np.fft.fft(P1)[:phaseLen])
        P2 = np.abs(np.fft.fft(P2)[:phaseLen])
        P3 = np.abs(np.fft.fft(P3)[:phaseLen])
        P2norm = P2/(np.sum(P1)+1e-17)
        fBin = fs/(2*phaseLen +1e-17)
        f750,f1k,f2k5 = int(-(-750//fBin)), int(-(-1000//fBin)), int(-(-2500//fBin))
        ratio =  np.sum(P2norm[f1k:f2k5]) / np.sum(P2norm[:f750])
        return np.ones((1,1))*ratio, names
    
    # Zero Crossing Rate
    def ZCR(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Zero_Crossing_Rate']
        fs,cough = data
        ZCR = (np.sum(np.multiply(cough[0:-1],cough[1:])<0)/(len(cough)-1))
        return np.ones((1,1))*ZCR, names
    
    # RMS Power
    def RMSP(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['RMS_Power']
        fs,cough = data
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*RMS, names
    
    # Dominant Frequency
    def DF(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Dominant_Freq']
        fs,cough = data
        cough_fortan = np.asfortranarray(cough)
        freqs, psd = signal.welch(cough_fortan)
        DF = freqs[np.argmax(psd)]
        return  np.ones((1,1))*DF, names
    
    def spectral_features(self, data):
        names = ["Spectral_Centroid","Spectral_Rolloff","Spectral_Spread","Spectral_Skewness","Spectral_Kurtosis","Spectral_Bandwidth"]
        fs, x = data
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) # positive frequencies
        sum_mag = np.sum(magnitudes)
        
        # spectral centroid = weighted mean of frequencies wrt FFT value at each frequency
        spec_centroid = np.sum(magnitudes*freqs) / sum_mag

        #spectral roloff = frequency below which 95% of signal energy lies
        cumsum_mag = np.cumsum(magnitudes)
        spec_rolloff = np.min(np.where(cumsum_mag >= 0.95*sum_mag)[0]) 

        #spectral spread = weighted standard deviation of frequencies wrt FFT value
        spec_spread = np.sqrt(np.sum(((freqs-spec_centroid)**2)*magnitudes) / sum_mag)

        #spectral skewness = distribution of the spectrum around its mean
        spec_skewness = np.sum(((freqs-spec_centroid)**3)*magnitudes) / ((spec_spread**3)*sum_mag)

        #spectral kurtosis = flatness of spectrum around its mean
        spec_kurtosis =  np.sum(((freqs-spec_centroid)**4)*magnitudes) / ((spec_spread**4)*sum_mag)

        #spectral bandwidth = weighted spectral standard deviation
        p=2
        spec_bandwidth = (np.sum(magnitudes*(freqs-spec_centroid)**p))**(1/p)

        return np.array([spec_centroid, spec_rolloff, spec_spread, spec_skewness, spec_kurtosis, spec_bandwidth]), names
    
    # Spectral Flatness and spectral standard deviation
    def SF_SSTD(self, data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of the feature
        names = ['Spectral_Flatness', 'Spectral_StDev']
        fs,sig = data
        nperseg = min(900,len(sig))
        noverlap = min(600,int(nperseg/2))
        freqs, psd = signal.welch(sig, fs, nperseg=nperseg, noverlap=noverlap)
        psd_len = len(psd)
        gmean = np.exp((1/psd_len)*np.sum(np.log(psd + 1e-17)))
        amean = (1/psd_len)*np.sum(psd)
        SF = gmean/amean
        SSTD = np.std(psd)
        return np.array([SF, SSTD]), names
        
    #Spectral Slope and Spectral Decrease
    def SSL_SD(self,data):
        names=['Spectral_Slope','Spectral_Decrease']
        b1=0
        b2=8000
        
        Fs, x = data
        s = np.absolute(np.fft.fft(x))
        s = s[:s.shape[0]//2]
        muS = np.mean(s)
        f = np.linspace(0,Fs/2,s.shape[0])
        muF = np.mean(f)

        bidx = np.where(np.logical_and(b1 <= f, f <= b2))
        slope = np.sum(((f-muF)*(s-muS))[bidx]) / np.sum((f[bidx]-muF)**2)

        k = bidx[0][1:]
        sb1 = s[bidx[0][0]]
        decrease = np.sum((s[k]-sb1)/(f[k]-1+1e-17)) / (np.sum(s[k]) + 1e-17)

        return np.array([slope, decrease]), names
    
    #MFCC
    def MFCC(self,data):
        # data: wav file of segment; fs, signal = wavfile.read(file)
        # output: value of MFCC coefficient
        names = []; names_mean = []; names_std = []
        fs, cough = data
        n_mfcc = 13
        for i in range(n_mfcc):
            names_mean = names_mean + ['MFCC_mean'+str(i)]
            names_std = names_std +  ['MFCC_std'+str(i)]
        names = names_mean + names_std
        mfcc = librosa.feature.mfcc(y = cough, sr = fs, n_mfcc = n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        mfcc = np.append(mfcc_mean,mfcc_std)
        return mfcc, names
    
    # Crest Factor
    def CF(self,data):
        """
        Compute the crest factor of the signal
        """
        fs, cough = data
        peak = np.amax(np.absolute(cough))
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*peak/RMS, ['Crest_Factor']
    
    def LGTH(self,data):
        "Compute the length of the segment in seconds"
        fs, cough = data
        return np.ones((1,1))*(len(cough)/fs), ['Cough_Length']
    
    # Power spectral Density 
    def PSD(self,data):
        feat = []
        fs,sig = data
        nperseg = min(900,len(sig))
        noverlap=min(600,int(nperseg/2))
        freqs, psd = signal.welch(sig, fs, nperseg=nperseg, noverlap=noverlap)
        dx_freq = freqs[1]-freqs[0]
        total_power = simpson(psd, dx=dx_freq)
        for lf, hf in self.FREQ_CUTS:
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            band_power = simpson(psd[idx_band], dx=dx_freq)
            feat.append(band_power/total_power)
        feat = np.array(feat)
        feat_names = [f'PSD_{lf}-{hf}' for lf, hf in self.FREQ_CUTS]
        return feat, feat_names


if __name__ == '__main__':
    main()