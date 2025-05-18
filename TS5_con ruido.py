# -*- coding: utf-8 -*-
"""
Actualizado: Incluye todas las señales procesadas con PSD por método de Welch.
@author: Valentina Aramayo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from scipy.io.wavfile import read as wavread
import pandas as pd

def vertical_flaten(a):
    return a.reshape(a.shape[0],1)

def analizar_psd_y_energia(señal, fs, titulo="PSD (Welch)"):
    """
    Calcula la PSD con Welch, grafica, y obtiene la frecuencia al 95% de energía acumulada.
    """
    señal_normalizada = señal / np.std(señal)
    N = len(señal)

    freqs, Pxx = sig.welch(señal_normalizada, fs, window='hamming', nperseg=N//20, nfft=N)

    energia_total = np.sum(Pxx)
    energia_acumulada = np.cumsum(Pxx) / energia_total
    indice_95 = np.where(energia_acumulada >= 0.98)[0][0]
    f_95 = freqs[indice_95]

    # Gráfico
    plt.figure()
    plt.plot(freqs, 10 * np.log10(Pxx + 1e-10), label='Welch (Hamming)')
    plt.axvline(x=f_95, color='r', linestyle='--', label=f'95% energía: {f_95:.2f} Hz')
    plt.axvline(f_98, color='green', linestyle='--', label='98% energía')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"{titulo} -> f_95 = {f_95:.2f} Hz")
    return f_95, energia_total, energia_acumulada

# ========= Procesamiento de señales ========= #

f_95_resultados = []
nombres = []

# -------- PPG sin ruido --------
ppg_sin_ruido = np.load('ppg_sin_ruido.npy')
f_95, _, _ = analizar_psd_y_energia(ppg_sin_ruido, fs=400, titulo="PPG sin ruido")
f_95_resultados.append(f_95)
nombres.append("PPG s/ruido")

# -------- PPG con ruido --------
ppg_con_ruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)
f_95, _, _ = analizar_psd_y_energia(ppg_con_ruido, fs=400, titulo="PPG con ruido")
f_95_resultados.append(f_95)
nombres.append("PPG c/ruido")

# -------- ECG sin ruido --------
ecg_sin_ruido = np.load('ecg_sin_ruido.npy')
f_95, _, _ = analizar_psd_y_energia(ecg_sin_ruido, fs=1000, titulo="ECG sin ruido")
f_95_resultados.append(f_95)
nombres.append("ECG s/ruido")

# -------- ECG con ruido --------
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_con_ruido = mat_struct['ecg_lead'].reshape(-1, 1) 
f_95, _, _ = analizar_psd_y_energia(ecg_con_ruido, fs=1000, titulo="ECG con ruido")
f_95_resultados.append(f_95)
nombres.append("ECG c/ruido")

# -------- Audio: La cucaracha --------
fs_audio1, audio1 = wavread('la cucaracha.wav')
f_95, _, _ = analizar_psd_y_energia(audio1, fs=fs_audio1, titulo="Audio: La cucaracha")
f_95_resultados.append(f_95)
nombres.append("La cucaracha")

# -------- Audio: Prueba PSD --------
fs_audio2, audio2 = wavread('prueba psd.wav')
f_95, _, _ = analizar_psd_y_energia(audio2, fs=fs_audio2, titulo="Audio: Prueba PSD")
f_95_resultados.append(f_95)
nombres.append("Prueba de audio")

# -------- Audio: Silbido --------
fs_audio3, audio3 = wavread('silbido.wav')
f_95, _, _ = analizar_psd_y_energia(audio3, fs=fs_audio3, titulo="Audio: Silbido")
f_95_resultados.append(f_95)
nombres.append("Silbido")

# ========= Tabla resumen ========= #

tabla_resultados = pd.DataFrame({
    "Ancho de Banda al 95% [Hz]": f_95_resultados
}, index=nombres)

tabla_resultados.style.set_caption("Estimación de Ancho de Banda al 95% de Energía con Welch") \
     .format("{:.2f}") \
     .set_table_styles([{
         "selector": "caption", 
         "props": [("font-size", "16px"), ("font-weight", "bold")]
     }])
