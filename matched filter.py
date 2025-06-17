from scipy.signal import find_peaks, correlate
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio
def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

# Cargar la señal ECG
fs= 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer las señales
ecg_one_lead = mat_struct['ecg_lead'].flatten()
hb1 = mat_struct['heartbeat_pattern1'].flatten()
hb2 = mat_struct['heartbeat_pattern2'].flatten()
qrs_pattern= mat_struct['qrs_pattern1'].flatten() #Una forma típica de latido (plantilla)
qrs_indices = mat_struct['qrs_detections'].flatten() #Índices temporales (en muestras)

# Paso 1: Normalización
ecg_filtrada_norm = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)
qrs_pattern_norm = (qrs_pattern - np.mean(qrs_pattern)) / np.std(qrs_pattern)

# Paso 2: Correlación normalizada
corr_norm = sig.correlate(ecg_filtrada_norm, qrs_pattern_norm, mode='same')

# Paso 3: Reescalado
ecg_rescaled = ecg_filtrada_norm / np.max(np.abs(ecg_filtrada_norm))
corr_rescaled = corr_norm / np.max(np.abs(corr_norm))

# %%
# Paso 4: Detección de picos en la correlación reescalada

threshold = 0.25  # Umbral relativo al máximo
peaks, properties = find_peaks(corr_rescaled, height=threshold, distance=200)
 
t_ecg=np.arange(len(corr_rescaled))/fs
# %%
# Paso 5: Gráfico conjunto

plt.figure(figsize=(14, 5))
plt.plot(t_ecg, ecg_rescaled, label='ECG (normalizado)', alpha=1, color='black')
plt.plot(t_ecg, corr_rescaled, label='Correlación normalizada (reescalada)', alpha=0.5, color='purple')
plt.plot(t_ecg[peaks], corr_rescaled[peaks], 'rx', label='Picos de correlación')
plt.title('ECG filtrado vs Correlación (con detección de picos)')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud escalada")
plt.grid(True)
plt.show()
# %%
# Paso 5: Gráfico conjunto cierto intervalo

plt.figure(figsize=(14, 5))
plt.plot(t_ecg, ecg_rescaled, label='ECG (normalizado)', alpha=0.7, color='black')
plt.plot(t_ecg, corr_rescaled, label='Correlación normalizada (reescalada)', alpha=1, color='purple')
plt.plot(t_ecg[peaks], corr_rescaled[peaks], 'bd', label='Picos de correlación')
plt.title('ECG filtrado vs Correlación (con detección de picos)')
plt.xlim([400, 410])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud escalada")
plt.legend()
plt.grid(True)
plt.show()
# %%
# Paso 5: Gráfico conjunto cierto intervalo

plt.figure(figsize=(14, 5))
plt.plot(t_ecg, ecg_rescaled, label='ECG (normalizado)', alpha=1, color='black')
plt.plot(t_ecg, corr_rescaled, label='Correlación normalizada (reescalada)', alpha=0.5, color='purple')
plt.plot(t_ecg[peaks], ecg_rescaled[peaks], 'bx', label='Detección de latido con matcher filter')
plt.plot(t_ecg[qrs_indices], ecg_rescaled[qrs_indices], 'bo', label='Detección de latido con qrs detections')
plt.title('ECG filtrado vs Correlación (con detección de picos)')
plt.xlim([400, 405])
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud escalada")
plt.legend()
plt.grid(True)
plt.show()
# %%
# Tolerancia en muestras para considerar que un latido fue detectado correctamente
tolerancia = 100 #100 muestras=100ms

# Contador de coincidencias
coincidencias = 0

# Recorremos los latidos detectados por correlación
for p in peaks:
    # Si hay al menos un qrs en el archivo cerca de este latido detectado, se cuenta como coincidencia
    if np.any(np.abs(qrs_indices - p) <= tolerancia):
        coincidencias += 1

# Proporción de detecciones correctas respecto al total de latidos del archivo
proporcion_detectados = coincidencias / len(qrs_indices)
print(f"Proporción de detecciones correctas (matcher vs archivo): {proporcion_detectados:.2%}")


# %%

# VER LATIDOS SUPERPUESTOS CON QRS DETECTADO EN MAT STRUCT
# Paso 6: Comparación de latidos superpuestos

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

# Supongamos que "segmentos_detectados" es tu array de latidos (cada fila es un latido)
ventana_central = slice(pre - 50, pre + 50)  # Por ejemplo, +/-50 muestras alrededor del centro

positivos = []
negativos = []

for latido in segmentos_detectados:
    centro = latido[ventana_central]
    max_abs = np.max(np.abs(centro))
    if np.max(centro) == max_abs:
        positivos.append(latido)
    else:
        negativos.append(latido)

positivos = np.array(positivos)
negativos = np.array(negativos)

# Ahora graficás separados
plt.figure(figsize=(12, 6))
for s in positivos:
    plt.plot(t, s, color='lightcoral', alpha=0.5)
for s in negativos:
    plt.plot(t, s, color='deepskyblue', alpha=0.5)

# Promedio general
plt.plot(t, np.mean(positivos, axis=0), color='red', label='Promedio Ventriculares', linewidth=2)
plt.plot(t, np.mean(negativos, axis=0), color='blue', label='Promedio Normales', linewidth=2)

plt.title("Latidos separados por polaridad")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (normalizada)")
plt.grid(True)
plt.legend()
plt.show()
# %%

