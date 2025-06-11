import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline
def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

# Lectura de ECG
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead']).flatten()
qrs_indices = mat_struct['qrs_detections'].flatten()
pre = 400  # muestras antes del QRS 
post = 300
segmentos = []
for idx in qrs_indices:
    if idx - pre >= 0 and idx + post< len(ecg_one_lead):
        segmento = ecg_one_lead[idx - pre :  idx+post]
        segmentos.append(segmento)

# Convertir a array 2D: cada fila es un segmento
segmentos_array = np.array(segmentos)

# Graficar los primeros 5 latidos superpuestos
plt.figure()
for i in range(5):
    plt.plot(segmentos_array[i], label=f'Segmento {i+1}')
plt.xlabel('Muestras (0 = QRS)')
plt.ylabel('Amplitud (uV)')
plt.grid(True)
plt.show()

#%%
# Parámetros de ventana alrededor del QRS
pre = 90  # muestras antes del QRS 
post = 20  # muestras 20ms despues de pre

# Extraer segmentos alrededor de los picos QRS
segmentos = []
for idx in qrs_indices:
    if idx - pre >= 0 and idx - pre + post < len(ecg_one_lead):
        segmento = ecg_one_lead[idx - pre :  idx - pre + post]
        segmentos.append(segmento)

# Convertir a array 2D: cada fila es un segmento
segmentos_array = np.array(segmentos)
# %% Estimacion de b en cada segmentos - grafico ECG ruidoso con linea de base 

# Calcular la media de cada segmento (fila)
medias_segmentos = np.mean(segmentos_array, axis=1)
x=qrs_indices - pre
cs = CubicSpline(x, medias_segmentos)
plt.plot(ecg_one_lead, color='pink')
plt.plot(x, cs(x), color='olive')
plt.title("Spline sobre la media de segmentos")
plt.grid(True)
plt.show()

# %% Resto linea de base al ECG 

cs = CubicSpline(x, medias_segmentos, extrapolate=True)
spline_full = cs(np.arange(len(ecg_one_lead)))
ECG_sin_b = ecg_one_lead - spline_full
plt.plot(ecg_one_lead, color='pink')
plt.plot(ECG_sin_b, color='teal')