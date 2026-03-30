import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig


# =========================================================
# PARAMETROS GENERALES
# =========================================================

fs = 200

canales = ["T7", "F8", "Cz", "P4"]

bandas = {
    "Delta": (1,4),
    "Theta": (4,8),
    "Alpha": (8,12),
    "Beta": (12,30)
}

# =========================================================
# FILTROS
# =========================================================

def notch_filter(signal, fs=200, f0=50, Q=30):

    b,a = sig.iirnotch(f0, Q, fs)

    return sig.filtfilt(b,a,signal)


def bandpass_filter(signal, fs=200, f1=1, f2=40):

    b,a = sig.butter(4,[f1,f2], btype="bandpass", fs=fs)

    return sig.filtfilt(b,a,signal)


# =========================================================
# PSD WELCH
# =========================================================

def psd_welch(signal, fs=200):

    f, psd = sig.welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=512,
        noverlap=256,
        scaling="density"
    )

    return f, psd


# =========================================================
# CARGAR EEG
# =========================================================

def cargar_eeg(archivo):

    datos = pd.read_csv(archivo)

    eeg = datos.iloc[:,1:5].values

    # convertir microvoltios a voltios
    eeg = eeg * 1e-6

    return eeg


# =========================================================
# PREPROCESAMIENTO
# =========================================================

def preprocesar_eeg(eeg):

    eeg_filtrado = np.zeros_like(eeg)

    for i in range(eeg.shape[1]):

        señal = eeg[:,i]

        señal = señal - np.mean(señal)

        señal = notch_filter(señal)

        señal = bandpass_filter(señal)

        eeg_filtrado[:,i] = señal

    return eeg_filtrado


# =========================================================
# ENERGIA POR BANDAS
# =========================================================

def energia_bandas(señal, fs=200):

    f, psd = psd_welch(señal, fs)

    energias = {}

    for banda,(f1,f2) in bandas.items():

        mask = (f>=f1) & (f<=f2)

        energia = np.trapezoid(psd[mask], f[mask])

        energias[banda] = energia

    return energias


# =========================================================
# ESPECTROGRAMA
# =========================================================

def espectrograma(señal, fs=200, titulo=""):

    f,t,Sxx = sig.spectrogram(
        señal,
        fs=fs,
        window="hann",
        nperseg=256,
        noverlap=128
    )

    mask = f <= 40

    plt.figure(figsize=(8,4))

    plt.pcolormesh(
        t,
        f[mask],
        10*np.log10(Sxx[mask] + 1e-12),
        shading="gouraud"
    )

    plt.ylabel("Frecuencia (Hz)")
    plt.xlabel("Tiempo (s)")
    plt.title(titulo)

    plt.colorbar(label="PSD (dB)")

    plt.show()


# =========================================================
# MATRIZ DE CORRELACION
# =========================================================

def matriz_correlacion(eeg):

    return np.corrcoef(eeg.T)


# =========================================================
# ANALISIS COMPLETO DE ARCHIVO
# =========================================================

def analizar_archivo(archivo):

    eeg = cargar_eeg(archivo)

    eeg_filtrado = preprocesar_eeg(eeg)

    resultados = []

    for i,canal in enumerate(canales):

        señal = eeg_filtrado[:,i]

        energias = energia_bandas(señal)

        energias["Canal"] = canal

        resultados.append(energias)

    df = pd.DataFrame(resultados)

    return df, eeg_filtrado


# =========================================================
# EJEMPLO DE USO
# =========================================================

archivo = "s01_ex05.txt"

df_resultados, eeg_filtrado = analizar_archivo(archivo)

print(df_resultados)

# =========================================================
# PSD EJEMPLO
# =========================================================

f,psd = psd_welch(eeg_filtrado[:,2])

plt.figure()

plt.plot(f,10*np.log10(psd))

plt.xlim(0,40)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (dB)")

plt.title("PSD Canal Cz")

plt.show()


# =========================================================
# ESPECTROGRAMA
# =========================================================

espectrograma(eeg_filtrado[:,2], titulo="Espectrograma Cz")


# =========================================================
# MATRIZ CORRELACION
# =========================================================

corr = matriz_correlacion(eeg_filtrado)

plt.figure(figsize=(6,5))

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar()

plt.xticks(range(4), canales)
plt.yticks(range(4), canales)

plt.title("Matriz de correlación EEG")

plt.show()