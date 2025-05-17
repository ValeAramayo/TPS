# -*- coding: utf-8 -*-
"""
Created on Sat May 10 19:29:13 2025

@author: Vale
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;

def procesar_ppg(ppg, fs=400, titulo="PSD de la señal PPG"):
    """
    Procesa una señal PPG: normaliza, calcula PSD por Blackman-Tukey, grafica y devuelve info clave.

    Retorna:
    - psd: densidad espectral de potencia
    - ff: eje de frecuencias
    - f_95: frecuencia donde se alcanza el 95% de la energía acumulada
    - energia_total: energía total del espectro
    - energia_acumulada: vector de energía acumulada (normalizada)
    """

    ppg = ppg / np.std(ppg)

    N = len(ppg)
    df = fs / N
    ff = np.linspace(0, fs, N, endpoint=False)

    psd = blackman_tukey(ppg, N // 10)

    # Gráfico
    plt.figure()
    plt.plot(ff[:N//2], 10 * np.log10(np.abs(psd[:N//2]) + 1e-10))
    plt.title(titulo)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [dB]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cálculo de energía acumulada
    psd_half = psd[:N//2].ravel()
    ff_half = ff[:N//2]

    energia_total = np.sum(psd_half)
    energia_acumulada = np.cumsum(psd_half) / energia_total
    indice_95 = np.where(energia_acumulada >= 0.95)[0][0]
    f_95 = ff_half[indice_95]

    print(f"Frecuencia al 95% de la energía acumulada: {f_95:.2f} Hz")

    return psd, ff, f_95, energia_total, energia_acumulada


#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
psd_cuca, ff_cuca, f_95_cuca, energia_tot_cuca, energia_acum_cuca = procesar_ppg(wav_data, fs_audio, titulo="la cucaracha")
fs_audio2, wav_data2 = sio.wavfile.read('prueba psd.wav')
psd_prueba, ff_prueba, f_95_prueba, energia_tot_prueba, energia_acum_prueba = procesar_ppg(wav_data2, fs_audio2, titulo="prueba de audio")
fs_audio3, wav_data3 = sio.wavfile.read('silbido.wav')
psd_silbido, ff_silbido, f_95_silbido, energia_tot_silbido, energia_acum_silbido= procesar_ppg(wav_data3, fs_audio3, titulo="Silbido")

from moviepy.editor import VideoFileClip

clip = VideoFileClip("WhatsApp Audio 2025-05-14 at 00.50.34.mp4")
audio = clip.audio
audio.write_audiofile("salida.wav")
fs_audio4, wav_data4 = sio.wavfile.read('WhatsApp Audio 2025-05-14 at 00.50.34.wav')
psd_vale_audi, ff_vale_audi, f_95__vale_audi, energia_tot_vale_audi, energia_acum_vale_audi= procesar_ppg(wav_data4, fs_audio4, titulo="Audio de vale - Las pastillas del abuelo")

# plt.figure()
# plt.plot(wav_data)
# plt.figure()
# plt.plot(wav_data2)
# plt.figure()
# plt.plot(wav_data3)

