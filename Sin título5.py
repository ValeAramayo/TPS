# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 13:05:02 2025

@author: Vale
"""

from scipy.signal import find_peaks, correlate
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Cargar la señal ECG
fs = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer las señales
ecg_one_lead = mat_struct['ecg_lead'].flatten()
hb1 = mat_struct['heartbeat_pattern1'].flatten()
hb2 = mat_struct['heartbeat_pattern2'].flatten()
qrs_indices = mat_struct['qrs_pattern1'].flatten().astype(int)  # importante convertir a enteros

# Paso 1: Normalización
ecg_filtrada_norm = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

# Paso 2: Correlación normalizada
qrs_pattern = hb1  # o hb2 si querés comparar con ese
qrs_pattern_norm = (qrs_pattern - np.mean(qrs_pattern)) / np.std(qrs_pattern)
corr_norm = sig.correlate(ecg_filtrada_norm, qrs_pattern_norm, mode='same')

# Paso 3: Reescalado
ecg_rescaled = ecg_filtrada_norm / np.max(np.abs(ecg_filtrada_norm))
corr_rescaled = corr_norm / np.max(np.abs(corr_norm))

# Paso 4: Detección de picos en la correlación
threshold = 0.25  # Umbral relativo al máximo
peaks, properties = find_peaks(corr_rescaled, height=threshold, distance=200)

# Paso 5: Comparación de latidos superpuestos

pre = 250  # muestras antes del QRS
post = 350  # muestras después
t = np.arange(-pre, post) * 1000 / fs  # eje en ms

def extraer_segmentos(indices, señal):
    segmentos = []
    for idx in indices:
        if idx - pre >= 0 and idx + post < len(señal):
            segmento = señal[idx - pre : idx + post]
            segmento -= np.mean(segmento)
            segmentos.append(segmento)
    return np.array(segmentos)

# 1. Latidos usando QRS del archivo
segmentos_mat = extraer_segmentos(qrs_indices, ecg_filtrada_norm)

plt.figure(figsize=(10,5))
for i in range(len(segmentos_mat)):
    plt.plot(t, segmentos_mat[i], color='lightblue', alpha=0.5)
plt.plot(t, np.mean(segmentos_mat, axis=0), color='blue', label='Promedio', linewidth=2)
plt.title('Latidos usando QRS de archivo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud (normalizada)')
plt.grid(True)
plt.legend()
plt.show()

# 2. Latidos usando detección propia (correlación)
segmentos_detectados = extraer_segmentos(peaks, ecg_filtrada_norm)

plt.figure(figsize=(10,5))
for i in range(len(segmentos_detectados)):
    plt.plot(t, segmentos_detectados[i], color='lightgreen', alpha=0.5)
plt.plot(t, np.mean(segmentos_detectados, axis=0), color='green', label='Promedio', linewidth=2)
plt.title('Latidos usando detección por correlación')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud (normalizada)')
plt.grid(True)
plt.legend()
plt.show()
