# =========================
# IMPORTS
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import welch

# =========================
# FUNCIONES DE CARGA
# =========================

def cargar_eeg(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    eeg = datos.iloc[9000:33000, 1:5].apply(pd.to_numeric).values

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
# FILTRO NUEVO (IIRDESIGN)
# =========================

def aplicar_filtro(eeg, fs):

    # Diseño automático Butterworth con especificaciones
    sos = sig.iirdesign(
        wp=[0.4, 40],      # banda pasante
        ws=[0.01, 45],    # banda de rechazo
        gpass=1,       # ripple permitido (dB)
        gstop=20,        # atenuación mínima (dB)
        ftype='butter',
        output='sos',
        fs=fs
    )

    # Filtrado sin fase
    eeg_filtrado = sig.sosfiltfilt(sos, eeg, axis=0)

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

path_crudo = "s01_ex01_s01.txt"
path_filtrado = "s01_ex01_s01.csv"

# CARGA
eeg_crudo, t = cargar_eeg(path_crudo, fs)
eeg_filtrado_archivo, _ = cargar_eeg_filtrado(path_filtrado, fs)

# FILTRADO NUEVO
eeg_filtrado = aplicar_filtro(eeg_crudo, fs)

print("Duración (s):", len(eeg_filtrado)/fs)

# =========================
# GRAFICO 1: CRUDO vs FILTRADO
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


# =========================
# GRAFICO 2: COMPARACIÓN FILTRADOS
# =========================

plt.figure(figsize=(12, 8))

for ch in range(4):
    plt.subplot(4,1,ch+1)

    plt.plot(t[:2000], eeg_filtrado_archivo[:2000, ch],
             label="Filtrado (archivo)", alpha=0.7)

    plt.plot(t[:2000], eeg_filtrado[1:2001, ch],
             linestyle='--', label="Filtrado (nuestro)")

    plt.title(f"Comparación filtrados - {canales[ch]}")
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
# PSD PROMEDIO
# =========================

f, psd_segs = psd_segmentos(segmentos, fs)

psd_mean = np.mean(psd_segs, axis=0)
psd_std = np.std(psd_segs, axis=0)

colores = ['b', 'r', 'g', 'm']

plt.figure(figsize=(10,6))

for ch in range(4):
    plt.semilogy(f, psd_mean[:, ch],
                 color=colores[ch],
                 label=canales[ch])

plt.title("PSD promedio (segmentos 4s)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 60)
plt.ylim(1e-6, 1e2)
plt.grid()
plt.legend()

plt.show()

# %%


# =========================
# PSD COMPARACIÓN
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
plt.xlim(0, 60)
plt.grid()
plt.legend()

plt.show()
# %%



# =========================
# VARIABILIDAD ENTRE SEGMENTOS
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