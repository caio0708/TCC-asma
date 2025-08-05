import numpy as np
import librosa
from scipy import signal
from scipy.integrate import simpson
import os
import sys
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import pickle
import argparse
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

class features:
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
        self.FREQ_CUTS = FREQ_CUTS
        self.n_PSD = len(FREQ_CUTS)
        
    def std_dev(self, data):
        names = ['std_dev']
        std_deviation = np.ones((1,1))*np.std(data[1])
        return std_deviation, names
    
    def dummy(self, data):
        names = ['dummy_feature_2','dummy_3']
        return np.array([1.,2.]), names
    
    def fft(self,data):
        fs, cough = data
        fftdata = np.fft.rfft(cough)
        return fftdata
    
    def EEPD(self, data):
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

    def PRE(self, data):
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
    
    def ZCR(self, data):
        names = ['Zero_Crossing_Rate']
        fs,cough = data
        ZCR = (np.sum(np.multiply(cough[0:-1],cough[1:])<0)/(len(cough)-1))
        return np.ones((1,1))*ZCR, names
    
    def RMSP(self, data):
        names = ['RMS_Power']
        fs,cough = data
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*RMS, names
    
    def DF(self, data):
        names = ['Dominant_Freq']
        fs,cough = data
        cough_fortan = np.asfortranarray(cough)
        freqs, psd = signal.welch(cough_fortan)
        DF = freqs[np.argmax(psd)]
        return  np.ones((1,1))*DF, names
    
    def spectral_features(self, data):
        names = ["Spectral_Centroid","Spectral_Rolloff","Spectral_Spread","Spectral_Skewness","Spectral_Kurtosis","Spectral_Bandwidth"]
        fs, x = data
        magnitudes = np.abs(np.fft.rfft(x))
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1])
        sum_mag = np.sum(magnitudes)
        
        spec_centroid = np.sum(magnitudes*freqs) / sum_mag
        cumsum_mag = np.cumsum(magnitudes)
        spec_rolloff = np.min(np.where(cumsum_mag >= 0.95*sum_mag)[0])
        spec_spread = np.sqrt(np.sum(((freqs-spec_centroid)**2)*magnitudes) / sum_mag)
        spec_skewness = np.sum(((freqs-spec_centroid)**3)*magnitudes) / ((spec_spread**3)*sum_mag)
        spec_kurtosis =  np.sum(((freqs-spec_centroid)**4)*magnitudes) / ((spec_spread**4)*sum_mag)
        p=2
        spec_bandwidth = (np.sum(magnitudes*(freqs-spec_centroid)**p))**(1/p)

        return np.array([spec_centroid, spec_rolloff, spec_spread, spec_skewness, spec_kurtosis, spec_bandwidth]), names
    
    def SF_SSTD(self, data):
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
    
    def MFCC(self,data):
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
    
    def CF(self,data):
        fs, cough = data
        peak = np.amax(np.absolute(cough))
        RMS = np.sqrt(np.mean(np.square(cough)))
        return np.ones((1,1))*peak/RMS, ['Crest_Factor']
    
    def LGTH(self,data):
        fs, cough = data
        return np.ones((1,1))*(len(cough)/fs), ['Cough_Length']
    
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

def classify_cough(x, fs, model, scaler):
    """Classify whether an inputted signal is a cough or not using filtering, feature extraction, and ML classification"""
    try: 
        x,fs = preprocess_cough(x,fs, cutoff=8000)  # Aumentado para 8000 Hz
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
        logging.debug(f"Probabilidade de tosse: {result[0]}")
        return result[0]
    except Exception as e:
        logging.error(f"Erro na extração de características: {e}")
        return 0

def preprocess_cough(x,fs, cutoff=8000, normalize=True, filter_=True, downsample=True):
    """Normalize, lowpass filter, and downsample cough samples"""
    fs_downsample = cutoff*2
    
    if len(x.shape)>1:
        x = np.mean(x,axis=1)
    if normalize:
        x = x/(np.max(np.abs(x))+1e-17)
    if filter_:
        b, a = butter(4, fs_downsample/fs, btype='lowpass')
        x = filtfilt(b, a, x)
    if downsample:
        x = signal.decimate(x, int(fs/fs_downsample))
    
    fs_new = fs_downsample
    return np.float32(x), fs_new

def detect_and_count_coughs(input_file, model, scaler, window_duration=0.5, threshold=0.5):
    fs, x = wavfile.read(input_file)
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
    
    total_samples = len(x)
    window_size = int(fs * window_duration)
    step_size = window_size // 2
    
    cough_count = 0
    cough_probs = []
    
    for start in range(0, total_samples - window_size + 1, step_size):
        segment = x[start:start + window_size]
        prob = classify_cough(segment, fs, model, scaler)
        cough_probs.append(prob)
        print(f"Janela {start//step_size}: Probabilidade = {prob}")
        if prob >= threshold:
            cough_count += 1
    
    print(f"Arquivo: {input_file}")
    print(f"Total de janelas analisadas: {len(cough_probs)}")
    print(f"Número de tosses detectadas (prob >= {threshold}): {cough_count}")
    return cough_count, cough_probs

def main(input_file):
    model_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classifier'
    scaler_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classification_scaler'

    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    count, _ = detect_and_count_coughs(input_file, model, scaler)
    return count

if __name__ == '__main__':
    audio_path = r"E:\Dev\TCC-asma\ia\uploads\audio\2d7e7a6a-bccc-4274-bddd-cf2b85ab1486.wav"
    model_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classifier'
    scaler_path = r'E:\Dev\TCC-asma\ia\model_artifacts\cough_classification_scaler'

    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    detect_and_count_coughs(audio_path, model, scaler)