 # -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:24:19 2025

@author: Vale
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import medfilt

def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)
# Definir kernel_size 
kernel_size = 201
kernel_size_2 = 1201
# Lectura de ECG
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead']).flatten()
# N = len(ecg_one_lead) 
ecg=ecg_one_lead[700000:745000]
medfilt(ecg, kernel_size=None)

# Aplicar filtro mediano ventana 200ms
ecg_filtradovgrande = medfilt(ecg, kernel_size=kernel_size)

# Aplicar filtro mediano ventana 600ms
ecg_filtradovchica=medfilt(ecg_filtradovgrande, kernel_size=kernel_size_2)

# Gráfica
plt.figure(figsize=(15, 6))
plt.plot(ecg, label='ECG Original', alpha=0.5)

plt.plot(ecg_filtradovchica, label='Linea de base', linewidth=1)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Comparación de ECG filtrado con mediana')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# Toda la senal 
# Aplicar filtro mediano ventana 200ms
ecg_filtradovgrande2 = medfilt(ecg_one_lead, kernel_size=kernel_size)

# Aplicar filtro mediano ventana 600ms
ecg_filtradovchica2=medfilt(ecg_filtradovgrande2, kernel_size=kernel_size_2)

# Gráfica
plt.figure(figsize=(15, 6))
plt.plot(ecg_one_lead, label='ECG Original', alpha=0.5)

plt.plot(ecg_filtradovchica2, label='Linea de base', linewidth=1)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Comparación de ECG filtrado con mediana')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# jugar con las ventanas, probar ppg q es un buen candidato