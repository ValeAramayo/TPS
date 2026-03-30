import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

# =========================
# FUNCIONES DE CARGA
# =========================

def cargar_eeg(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    # 24000 muestras exactas → 30 segmentos de 4s
    eeg = datos.iloc[3000: 27000, 1:5].apply(pd.to_numeric).values

    eeg -= np.mean(eeg, axis=0)
    tiempo = np.arange(len(eeg)) / fs

    return eeg, tiempo


def cargar_eeg_filtrado(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    eeg_raw = datos.iloc[:, 1:5].apply(pd.to_numeric)

    # [p4, cz, F8, T7] → [T7, F8, Cz, P4]
    orden_columnas = [3, 2, 1, 0]
    eeg = eeg_raw.iloc[:, orden_columnas].values

    eeg -= np.mean(eeg, axis=0)
    tiempo = np.arange(len(eeg)) / fs

    return eeg, tiempo


# =========================
# FILTROS
# =========================

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = fs / 2
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')


def aplicar_filtro(eeg, fs):

    # Bandpass 1–40 Hz
    b, a = butter_bandpass(1, 40, fs, order=1)
    eeg_filtrado = filtfilt(b, a, eeg, axis=0)

    # Notch 50 Hz
    w0 = 50 / (fs/2)
    b_notch, a_notch = iirnotch(w0, Q=30)
    eeg_filtrado = filtfilt(b_notch, a_notch, eeg_filtrado, axis=0)

    return eeg_filtrado


# =========================
# SEGMENTACIÓN
# =========================

def segmentar(eeg, fs, duracion=4):
    muestras_segmento = int(fs * duracion)
    total_segmentos = len(eeg) // muestras_segmento

    segmentos = []

    for i in range(total_segmentos):
        inicio = i * muestras_segmento
        fin = inicio + muestras_segmento
        segmentos.append(eeg[inicio:fin, :])

    return np.array(segmentos)


# =========================
# PSD POR SEGMENTO
# =========================

def psd_segmentos(segmentos, fs):
    psd_list = []

    for seg in segmentos:
        f, Pxx = welch(seg, fs=fs, nperseg=len(seg), axis=0)
        psd_list.append(Pxx)

    return f, np.array(psd_list)


# =========================
# MAIN
# =========================

fs = 200

path_crudo = "s01_ex02_s01.txt"
path_filtrado = "s01_ex02_s01.csv"

# CARGA
eeg_crudo, t = cargar_eeg(path_crudo, fs)
eeg_filtrado_archivo, _ = cargar_eeg_filtrado(path_filtrado, fs)

# FILTRADO PROPIO
eeg_filtrado = aplicar_filtro(eeg_crudo, fs)

print("Duración (s):", len(eeg_filtrado)/fs)

# =========================
# GRAFICO 1: CRUDO vs FILTRADO (COMPARACIÓN DE MI TRABAJO Y LA MAESTRIA QUE USA LOS DATOS)
# =========================

canales = ["T7", "F8", "Cz", "P4"]

plt.figure(figsize=(12, 8))

for ch in range(4):
    plt.subplot(4,1,ch+1)

    plt.plot(t[:2000], eeg_crudo[:2000, ch], label="Crudo", alpha=0.6)
    plt.plot(t[:2000], eeg_filtrado[:2000, ch], 
             label="Filtrado", linestyle='--')

    plt.title(f"Crudo vs Filtrado - {canales[ch]}")
    plt.grid()

    if ch == 0:
        plt.legend()

plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()
# %%


# =========================
# GRAFICO 1: CRUDO vs FILTRADO (POR MI)
# =========================

ch = 1  # canal que quieras

plt.figure(figsize=(10, 6))

# =========================
# ARRIBA: SEÑAL CRUDA
# =========================
plt.subplot(2,1,1)
plt.plot(t[2000:3000], eeg_crudo[2000:3000, ch], color='black')
plt.title("Señal Original")
plt.grid()
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
# =========================
# ABAJO: SEÑAL FILTRADA
# =========================
plt.subplot(2,1,2)
plt.plot(t[2000:3000], eeg_filtrado[2000:3000, ch], color='black', linestyle='--')
plt.title("Señal Filtrada")
plt.grid()
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.tight_layout()
plt.show()
# %%


# ========================= 
# GRAFICO 2: FILTRADO vs FILTRADO (TIEMPO) 
# ========================= 
plt.figure(figsize=(12, 8)) 
for ch in range(4): 
    plt.subplot(4,1,ch+1) 
    plt.plot(t[:2000], eeg_filtrado_archivo[:2000, ch], label="Filtrado (archivo)", alpha=0.7) 
    plt.plot(t[:2000], eeg_filtrado[1:2001, ch], linestyle='--', label="Filtrado (nuestro)") 
    plt.title(f"Comparación filtrados - Canal {canales[ch]}") 
    plt.ylabel("Amplitud") 
    plt.grid() 
    if ch == 0: 
        plt.legend() 
        plt.xlabel("Tiempo (s)") 
        plt.tight_layout() 
        plt.show()

# =========================
# SEGMENTACIÓN
# =========================

segmentos = segmentar(eeg_filtrado, fs, duracion=4)
print("Cantidad de segmentos:", len(segmentos))


# =========================
# PSD POR SEGMENTOS
# =========================

f, psd_segs = psd_segmentos(segmentos, fs)

psd_mean = np.mean(psd_segs, axis=0)
psd_std = np.std(psd_segs, axis=0)

colores = ['b', 'r', 'g', 'm']


# =========================
# GRAFICO 3: PSD PROMEDIO
# =========================

plt.figure(figsize=(10,6))

for ch in range(4):
    plt.semilogy(f, psd_mean[:, ch], 
                 color=colores[ch], 
                 label=canales[ch])

plt.title("PSD promedio (segmentos 4s sin overlap)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 100)
plt.ylim(1e-8, 1e2)
plt.grid()
plt.legend()

plt.show()



# =========================
# GRAFICO 4: COMPARACIÓN FILTRADOS (PSD)
# =========================

f, Pxx_arch = psd_segmentos(segmentar(eeg_filtrado_archivo, fs), fs)
Pxx_arch_mean = np.mean(Pxx_arch, axis=0)

plt.figure(figsize=(10,6))

for ch in range(4):
    plt.semilogy(f, Pxx_arch_mean[:, ch], 
                 label=f"{canales[ch]} archivo", alpha=0.7)

    plt.semilogy(f, psd_mean[:, ch], 
                 linestyle='--', label=f"{canales[ch]} nuestro")

plt.title("Comparación PSD filtrado")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 100)
plt.ylim(1e-8, 1e2)
plt.grid()
plt.legend()

plt.show()

# %%

# =========================
# GRAFICO 5: ALGUNOS SEGMENTOS (VER VARIABILIDAD)
# =========================

plt.figure(figsize=(10,6))

for i in range(min(5, len(psd_segs))):
    plt.semilogy(f, psd_segs[i, :, 0], alpha=0.5)

plt.title("PSD de segmentos individuales (Canal T7)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 60)
plt.grid()

plt.show()