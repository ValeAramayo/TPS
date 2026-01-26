# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 19:32:54 2026

@author: Vale
"""

bandas = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30)
}

colores_bandas = {
    "Delta": "tab:blue",
    "Theta": "tab:orange",
    "Alpha": "tab:green",
    "Beta": "tab:red"
}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# =========================================================
# Funciones (las mismas que ya usás)
# =========================================================

def blackman_tukey(x, M=None):
    x = x.ravel()
    N = len(x)

    if M is None:
        M = N // 10

    r_len = 2 * M - 1
    xx = x[:r_len]

    r = np.correlate(xx, xx, mode='same') / r_len
    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n=N))

    return Px


def procesar_señal(señal, fs, titulo="PSD de la señal"):
    señal = señal / np.std(señal)
    N = len(señal)

    ff = np.linspace(0, fs, N, endpoint=False)
    psd = blackman_tukey(señal)

    psd_half = psd[:N // 2]
    ff_half = ff[:N // 2]

    energia_total = np.sum(psd_half)
    energia_acumulada = np.cumsum(psd_half) / energia_total

    f_95 = ff_half[np.where(energia_acumulada >= 0.95)[0][0]]
    f_98 = ff_half[np.where(energia_acumulada >= 0.98)[0][0]]

    # Gráfico
    plt.figure()
    plt.plot(ff_half, 10 * np.log10(psd_half + 1e-10))
    plt.axvline(f_95, color='r', linestyle='--', label=f'95%: {f_95:.2f} Hz')
    plt.axvline(f_98, color='g', linestyle='--', label=f'98%: {f_98:.2f} Hz')
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return psd, ff, f_95, f_98, energia_total, energia_acumulada
def procesar_bandas(señal, fs, titulo="Energía por bandas"):
    señal = señal / np.std(señal)
    N = len(señal)

    ff = np.linspace(0, fs, N, endpoint=False)
    psd = blackman_tukey(señal)

    psd_half = psd[:N // 2]
    ff_half = ff[:N // 2]

    energia_total = np.sum(psd_half)
    energia_bandas = {}

    for banda, (fmin, fmax) in bandas.items():
        idx = (ff_half >= fmin) & (ff_half < fmax)
        energia_bandas[banda] = np.sum(psd_half[idx]) / energia_total * 100

    # Gráfico PSD + bandas
    plt.figure()
    plt.plot(ff_half, 10*np.log10(psd_half + 1e-12), color="black")

    for banda, (fmin, fmax) in bandas.items():
        plt.axvspan(
            fmin, fmax,
            color=colores_bandas[banda],
            alpha=0.25,
            label=banda
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(dict(zip(labels, handles)).values(),
               dict(zip(labels, handles)).keys())

    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return energia_bandas

# =========================================================
# Lectura EEG OpenBCI
# =========================================================

datos = pd.read_csv(
    r"C:\Users\Vale\Documents\APS\APS_vale\TPS\s01_ex05.txt",
    sep=",",
    skiprows=5
)

eeg = datos.iloc[:, 1:5]
eeg = eeg.apply(pd.to_numeric).values

fs = 200  # Hz

# Quitar offset
eeg = eeg - np.mean(eeg, axis=0)

# Tiempo
n = eeg.shape[0]
tiempo = np.arange(n) / fs

# Ventana de 2 segundos
muestras = int(2 * fs)
eeg = eeg[:muestras]
tiempo = tiempo[:muestras]

# =========================================================
# Gráfica temporal
# =========================================================

canales = ["T7", "F8", "Cz", "P4"]

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

for i in range(4):
    ax[i].plot(tiempo, eeg[:, i])
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].set_ylabel("Amplitud")
    ax[i].grid(True)

ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()

# =========================================================
# PSD + Ancho de banda (GUARDADO SIN PISAR)
# =========================================================

psd_eeg = {}
ff_eeg = {}
f95_eeg = {}
f98_eeg = {}
energia_tot_eeg = {}
energia_acum_eeg = {}

for i, canal in enumerate(canales):
    psd, ff, f95, f98, Etot, Eacc = procesar_señal(
        eeg[:, i],
        fs=fs,
        titulo=f"PSD EEG – Canal {canal}"
    )

    psd_eeg[canal] = psd
    ff_eeg[canal] = ff
    f95_eeg[canal] = f95
    f98_eeg[canal] = f98
    energia_tot_eeg[canal] = Etot
    energia_acum_eeg[canal] = Eacc

# =========================================================
# Tabla resumen de anchos de banda
# =========================================================

tabla_eeg = pd.DataFrame({
    "Ancho de banda 95% [Hz]": f95_eeg,
    "Ancho de banda 98% [Hz]": f98_eeg
})

print("\nAncho de banda EEG:")
print(tabla_eeg)

# =========================================================
# DISEÑO DEL FILTRO (MISMO PARA TODOS)
# =========================================================

fpass = [1, 40]      # banda EEG
fstop = [0.1, 50]
ripple = 0.5
atenuacion = 40

sos = sig.iirdesign(
    fpass, fstop,
    ripple, atenuacion,
    ftype='butter',
    output='sos',
    fs=fs
)

# =========================================================
# FILTRADO DE SEÑALES COMPLETAS
# =========================================================

eeg_filtrado = np.zeros_like(eeg)

for i in range(4):
    eeg_filtrado[:, i] = sig.sosfilt(sos, eeg[:, i])

# =========================================================
# GRÁFICA TEMPORAL (SEÑAL COMPLETA FILTRADA)
# =========================================================

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

for i in range(4):
    ax[i].plot(tiempo, eeg_filtrado[:, i])
    ax[i].set_title(f"Canal {canales[i]} – Filtrado")
    ax[i].grid(True)

ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()

# =========================================================
# PSD + Ancho de banda de la senial filtrada (para ver como se atenuan los 50 hz)
# =========================================================

psd_eeg = {}
ff_eeg = {}
f95_eeg = {}
f98_eeg = {}
energia_tot_eeg = {}
energia_acum_eeg = {}

for i, canal in enumerate(canales):
    psd, ff, f95, f98, Etot, Eacc = procesar_señal(
    eeg_filtrado[:, i],
    fs=fs,
    titulo=f"PSD EEG FILTRADA – Canal {canal}"
)


    psd_eeg[canal] = psd
    ff_eeg[canal] = ff
    f95_eeg[canal] = f95
    f98_eeg[canal] = f98
    energia_tot_eeg[canal] = Etot
    energia_acum_eeg[canal] = Eacc

 # =========================================================
 # Estimacion de energia por bandas 
 # =========================================================   
energias_eeg = {}

for i, canal in enumerate(canales):
    energias = procesar_bandas(
        eeg_filtrado[:, i],
        fs=fs,
        titulo=f"Energía por bandas – {canal}"
    )
    energias_eeg[canal] = energias

tabla_bandas = pd.DataFrame(energias_eeg).T
print(tabla_bandas)



