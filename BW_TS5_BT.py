# -*- coding: utf-8 -*-
"""
Created on Sat May 10 10:59:23 2025

@author: Usuario
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write, read
import pandas as pd

def vertical_flaten(a):
    return a.reshape(a.shape[0],1)

def blackman_tukey(x,  M = None):    
    x_z = x.shape
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # Aplanar los arrays para np.correlate.
    xx = x.ravel()[:r_len]

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )
    Px = Px.reshape(x_z)

    return Px;

def procesar_señal(señal, fs, titulo="PSD de la señal", es_silbido=False):
    """
    Procesa una señal: normaliza, calcula PSD por Blackman-Tukey, grafica y devuelve info clave.
    Parámetros:
    - señal: array con la señal a analizar
    - fs: frecuencia de muestreo
    - titulo: título para el gráfico
    - es_silbido: si es True, usa M = N//4 en Blackman-Tukey para mejor resolución en tonos puros
    """

    señal = señal / np.std(señal)
    N = len(señal)

    if es_silbido:
        M = N // 4
    else:
        M = N // 10

    df = fs / N
    ff = np.linspace(0, fs, N, endpoint=False)

    psd = blackman_tukey(señal, M)

    # Cálculo de energía acumulada
    psd_half = psd[:N//2].ravel()
    ff_half = ff[:N//2]

    energia_total = np.sum(psd_half)
    energia_acumulada = np.cumsum(psd_half) / energia_total
    indice_95 = np.where(energia_acumulada >= 0.95)[0][0]
    f_95 = ff_half[indice_95]
    indice_98 = np.where(energia_acumulada >= 0.98)[0][0]
    f_98 = ff_half[indice_98]

    # Gráfico
    plt.plot(ff[:N//2], 10 * np.log10(np.abs(psd[:N//2]) + 1e-10))
    plt.axvline(x=f_95, color='r', linestyle='--', label=f'95% energía: {f_95:.2f} Hz')
    plt.axvline(f_98, color='green', linestyle='--', label='98% energía')
    plt.title(titulo)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [dB]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Frecuencia al 95% de la energía acumulada: {f_95:.2f} Hz")
    print(f"Frecuencia al 98% de la energía acumulada: {f_98:.2f} Hz")

    return psd, ff, f_95, f_98, energia_total, energia_acumulada

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido##
##################

ppg_c_ruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
psd_c, ff_c, f_95_c_ppg, f_98_c_ppg, energia_tot_c, energia_acum_c = procesar_señal(ppg_c_ruido, fs=400, titulo="PPG con ruido")

##################
## PPG sin ruido
##################

ppg_s_ruido = np.load('ppg_sin_ruido.npy')
psd_s_ppg, ff_s_ppg, f_95_s_ppg, f_98_s_ppg, energia_tot_s_ppg, energia_acum_s_ppg = procesar_señal(ppg_s_ruido, fs=400, titulo="PPG sin ruido")

##################
## ECG con ruido
##################

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_c_ruido = vertical_flaten(mat_struct['ecg_lead'])
ecg_c_ruido = mat_struct['ecg_lead'].reshape(-1, 1)

psd_c_ecg, ff_c_ecg, f_95_c_ecg, f_98_c_ecg, energia_tot_c_ecg, energia_acum_c_ecg = procesar_señal(ecg_c_ruido, fs=1000, titulo="ECG con ruido")

##################
## ECG sin ruido
##################

ecg_s_ruido = np.load('ecg_sin_ruido.npy')
psd_s_ecg, ff_s_ecg, f_95_s_ecg, f_98_s_ecg, energia_tot_s_ecg, energia_acum_s_ecg = procesar_señal(ecg_s_ruido, fs=1000, titulo="ECG sin ruido")

####################
# Lectura de audio #
####################

fs_audio, wav_data = read('la cucaracha.wav')
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]
wav_data = wav_data.astype(np.float64)
psd_cuca, ff_cuca, f_95_cuca, f_98_cuca, energia_tot_cuca, energia_acum_cuca = procesar_señal(wav_data, fs_audio, titulo="La cucaracha")

fs_audio2, wav_data2 = read('prueba psd.wav')
if wav_data2.ndim > 1:
    wav_data2 = wav_data2[:, 0]
wav_data2 = wav_data2.astype(np.float64)
psd_prueba, ff_prueba, f_95_prueba, f_98_prueba, energia_tot_prueba, energia_acum_prueba = procesar_señal(wav_data2, fs_audio2, titulo="Prueba de audio")

fs_audio3, wav_data3 = read('silbido.wav')
if wav_data3.ndim > 1:
    wav_data3 = wav_data3[:, 0]
wav_data3 = wav_data3.astype(np.float64)
psd_silbido, ff_silbido, f_95_silbido, f_98_silbido, energia_tot_silbido, energia_acum_silbido= procesar_señal(wav_data3, fs_audio3, titulo="Silbido", es_silbido=True)

# Crear tabla de resultados
tabla3 = pd.DataFrame({
    "Ancho de Banda al 95% [Hz]": [
        f_95_c_ecg,
        f_95_s_ecg,
        f_95_c_ppg,
        f_95_s_ppg,
        f_95_cuca,
        f_95_prueba,
        f_95_silbido
    ],
    "Ancho de Banda al 98% [Hz]": [
        f_98_c_ppg,
        f_98_s_ppg,
        f_98_c_ecg,
        f_98_s_ecg,
        f_98_cuca,
        f_98_prueba,
        f_98_silbido
    ]
}, index=[
    "ECG c/ruido",
    "ECG s/ruido",
    "PPG c/ruido",
    "PPG s/ruido",
    "La cucaracha",
    "Prueba de audio",
    "Silbido"
])

# Estilo para la tabla
tabla3.style.set_caption("Estimación de Ancho de Banda al 95% y 98% de Energía") \
     .format("{:.2f}") \
     .set_table_styles([{
         "selector": "caption",
         "props": [("font-size", "16px"), ("font-weight", "bold")]
     }])
