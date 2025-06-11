# -- coding: utf-8 --
"""
Created on Wed Jun  4 21:30:49 2025

@author: Vale
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

# Lectura de ECG
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead']).flatten()
ecg_one_lead= (ecg_one_lead-np.mean(ecg_one_lead))/ np.std (ecg_one_lead)

qrs_indices = mat_struct['qrs_detections'].flatten()

pre = 250  # muestras antes del QRS 
post = 350
segmentos = []
for idx in qrs_indices:
    if idx - pre >= 0 and idx + post< len(ecg_one_lead):
        segmento = ecg_one_lead[idx - pre :  idx+post] -np.mean(ecg_one_lead[idx - pre :  idx+post])
        segmentos.append(segmento)

# Convertir a array 2D: cada fila es un segmento
segmentos_array = np.array(segmentos)
M= len(segmentos_array)
# Graficar todos los latidos superpuestos
plt.figure()
for i in range(M):
    plt.plot(segmentos_array[i])
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud (uV)')
plt.grid(True)
plt.show()