
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: Valentina Aramayo
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


#################
#Lectura de ECG #
#################

# Suponiendo que ya cargaste el archivo:
fs_ecg = 1000  # Hz
import scipy.io as sio

# Cargar el archivo .mat correctamente
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer la señal de ECG
ecg = mat_struct['ecg_lead'].reshape(-1, 1)  # Asegúrate de usar la variable correcta que contiene el ECG

# plt.figure()
# plt.plot(ecg)

# Normalización por varianza
ecg= ecg[:12000]
ecg = ecg / np.std(ecg)

# Aplicar la función Blackman-Tukey
N_ecg = ecg.shape[0]
df_ecg = fs_ecg / N_ecg
ff_ecg = np.linspace(0, fs_ecg, N_ecg, endpoint=False)

# Estimación de PSD (Blackman-Tukey con autocorrelación recortada)
psd_bt = blackman_tukey(ecg, N_ecg // 10)

# Graficar la PSD (en dB)
plt.figure()
plt.plot(ff_ecg[:N_ecg//2], 10 * np.log10(np.abs(psd_bt[:N_ecg//2]) + 1e-10))
plt.title('PSD de la señal ECG - Método Blackman-Tukey')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral [dB]')
plt.grid(True)
plt.show()

# Verificar que se cumpla parseval luego estiar Bw plantear un proporcion de 95 o 98% del area y ver con que fre
# frecuencia coincide la x 


#%%

# Usamos solo la mitad positiva del espectro
psd_half = psd_bt[:N_ecg//2].ravel()
frequencies = ff_ecg[:N_ecg//2]

# Energía total 
energia_total = np.sum(psd_half) 

# Energía acumulada
energia_acumulada = np.cumsum(psd_half) / energia_total

# Buscamos la frecuencia donde la energía acumulada supera el 95%

indice_95 = np.where(energia_acumulada >= 0.95)[0][0]
f_95 = frequencies[indice_95]
print(f"Frecuencia al 95% de la energía acumulada: {f_95:.2f} Hz")




