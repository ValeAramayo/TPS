import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Lectura del archivo OpenBCI
# =========================
datos = pd.read_csv(
    r"C:\Users\Vale\Documents\APS\APS_vale\TPS\s01_ex01_s01.txt",
    sep=",",
    skiprows=5
)

# =========================
# Canales EEG (EXG)
# =========================
eeg = datos.iloc[:, 1:5]
eeg = eeg.apply(pd.to_numeric).values

# =========================
# Parámetros
# =========================
fs = 200  # Hz

# =========================
# Quitar offset
# =========================
eeg = eeg - np.mean(eeg, axis=0)

# =========================
# Tiempo
# =========================
n = eeg.shape[0]
tiempo = np.arange(n) / fs

# =========================
# Ventana corta
# =========================
muestras = int(2 * fs)
eeg = eeg[:muestras]
tiempo = tiempo[:muestras]

# =========================
# Gráfica
# =========================
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
canales = ["T7", "F8", "Cz", "P4"]

for i in range(4):
    ax[i].plot(tiempo, eeg[:, i])
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].set_ylabel("Amplitud")
    ax[i].grid(True)

ax[-1].set_xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()

##################
## Primer sujeto ##
##################

ppg_c_ruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
psd_c, ff_c, f_95_c_ppg, f_98_c_ppg, energia_tot_c, energia_acum_c = procesar_señal(ppg_c_ruido, fs=400, titulo="PPG con ruido")
