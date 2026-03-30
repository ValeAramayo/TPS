# -*- coding: utf-8 -*-
"""
Procesamiento EEG generalizado
Primera persona: análisis completo
Resto: solo filtrado + energía por bandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# =========================================================
# DEFINICIÓN DE BANDAS
# =========================================================

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

canales = ["T7", "F8", "Cz", "P4"]
fs = 200  # Hz

# =========================================================
# FUNCIONES
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


def procesar_señal(señal, fs, titulo):
    señal = señal / np.std(señal)
    N = len(señal)

    ff = np.linspace(0, fs, N, endpoint=False)
    psd = blackman_tukey(señal)

    psd_half = psd[:N // 2]
    ff_half = ff[:N // 2]

    energia_total = np.sum(psd_half)
    energia_acum = np.cumsum(psd_half) / energia_total

    f95 = ff_half[np.where(energia_acum >= 0.95)[0][0]]
    f98 = ff_half[np.where(energia_acum >= 0.98)[0][0]]

    plt.figure()
    plt.plot(ff_half, 10*np.log10(psd_half + 1e-12))
    plt.axvline(f95, color='r', linestyle='--', label=f'95%: {f95:.2f} Hz')
    plt.axvline(f98, color='g', linestyle='--', label=f'98%: {f98:.2f} Hz')
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return f95, f98


def procesar_bandas_con_plot(señal, fs, titulo):
    señal = señal / np.std(señal)
    N = len(señal)

    ff = np.linspace(0, fs, N, endpoint=False)
    psd = blackman_tukey(señal)

    psd_half = psd[:N // 2]
    ff_half = ff[:N // 2]

    Etot = np.sum(psd_half)
    energias = {}

    plt.figure()
    plt.plot(ff_half, 10*np.log10(psd_half + 1e-12), color="black")

    for banda, (fmin, fmax) in bandas.items():
        idx = (ff_half >= fmin) & (ff_half < fmax)
        energias[banda] = np.sum(psd_half[idx]) / Etot * 100
        plt.axvspan(fmin, fmax, color=colores_bandas[banda], alpha=0.25, label=banda)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(dict(zip(labels, handles)).values(),
               dict(zip(labels, handles)).keys())

    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return energias


def energia_bandas_sin_plot(señal, fs):
    señal = señal / np.std(señal)
    N = len(señal)

    ff = np.linspace(0, fs, N, endpoint=False)
    psd = blackman_tukey(señal)

    psd_half = psd[:N // 2]
    ff_half = ff[:N // 2]

    Etot = np.sum(psd_half)

    return {
        banda: np.sum(psd_half[(ff_half >= fmin) & (ff_half < fmax)]) / Etot * 100
        for banda, (fmin, fmax) in bandas.items()
    }


def cargar_eeg(path, fs, ventana_seg=2):
    datos = pd.read_csv(path, sep=",", skiprows=5)
    eeg = datos.iloc[:, 1:5].apply(pd.to_numeric).values
    eeg -= np.mean(eeg, axis=0)

    muestras = int(ventana_seg * fs)
    eeg = eeg[:muestras]
    tiempo = np.arange(muestras) / fs

    return eeg, tiempo


# =========================================================
# DISEÑO DEL FILTRO (ÚNICO)
# =========================================================
sos = sig.iirdesign(
    [1, 40],      # wp
    [0.1, 50],    # ws
    0.5,          # gpass
    40,           # gstop
    ftype='butter',
    output='sos',
    fs=fs
)


# =========================================================
# ARCHIVOS
# =========================================================

base_path = r"C:\Users\Vale\Documents\APS\APS_vale\TPS"

personas = {
    "P1": ["s01_ex01_s01.txt", "s01_ex05.txt", "s01_ex06.txt", "s01_ex07.txt"],
    "P2": ["s02_ex01_s01.txt", "s02_ex05.txt", "s02_ex06.txt", "s02_ex07.txt"],
   #  "P3": ["s03_ex01_s01.txt", "s03_ex01_s02.txt", "s03_ex01_s03.txt", "s03_ex01_s04.txt"],
    # "P4": ["s04_ex01_s01.txt", "s04_ex01_s02.txt", "s04_ex01_s03.txt", "s04_ex01_s04.txt"],
   #  "P5": ["s05_ex01_s01.txt", "s05_ex01_s02.txt", "s05_ex01_s03.txt", "s05_ex01_s04.txt"],
}

# %%
# =========================================================

# ===== PRIMERA PERSONA – ANÁLISIS COMPLETO =====
# =========================================================

persona_ref = "P1"
archivo_ref = personas[persona_ref][0]

eeg, tiempo = cargar_eeg(f"{base_path}\\{archivo_ref}", fs)

# Gráfica temporal
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for i in range(4):
    ax[i].plot(tiempo, eeg[:, i])
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].grid(True)
ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()

# Filtrado
eeg_filtrado = np.zeros_like(eeg)
for i in range(4):
    eeg_filtrado[:, i] = sig.sosfilt(sos, eeg[:, i])

# PSD + ancho de banda
for i, canal in enumerate(canales):
    procesar_señal(eeg[:, i], fs, f"PSD EEG – {canal}")
    procesar_señal(eeg_filtrado[:, i], fs, f"PSD EEG FILTRADA – {canal}")

# Energía por bandas (con plots)
for i, canal in enumerate(canales):
    procesar_bandas_con_plot(
        eeg_filtrado[:, i],
        fs,
        f"Energía por bandas – {canal}"
    )

# %%
# =========================================================

# ===== RESTO DE PERSONAS – SOLO PROCESAMIENTO =====
# =========================================================

resultados = {}

for persona, archivos in personas.items():
    for archivo in archivos:

        if persona == persona_ref and archivo == archivo_ref:
            continue

        eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)

        eeg_filtrado = np.zeros_like(eeg)
        for i in range(4):
            eeg_filtrado[:, i] = sig.sosfilt(sos, eeg[:, i])

        resultados[(persona, archivo)] = {
            canal: energia_bandas_sin_plot(eeg_filtrado[:, i], fs)
            for i, canal in enumerate(canales)
        }

# =========================================================
# TABLA FINAL
# =========================================================

tabla_final = pd.concat(
    {k: pd.DataFrame(v).T for k, v in resultados.items()},
    names=["Persona", "Archivo"]
)

print(tabla_final)
# %%
# =========================================================

# ===== CONSTRUIR DATAFRAME LARGO PARA ANÁLISIS =====
# =========================================================

rows = []

for (persona, archivo), datos_canales in resultados.items():
    for canal, bandas_dict in datos_canales.items():
        row = {
            "Persona": persona,
            "Archivo": archivo,
            "Canal": canal,
            **bandas_dict
        }
        rows.append(row)

df_long = pd.DataFrame(rows)

# Extraer situación desde el nombre del archivo
def extraer_situacion(nombre):
    if "ex05" in nombre:
        return "Situación 1"
    elif "ex06" in nombre:
        return "Situación 2"
    elif "ex07" in nombre:
        return "Situación 3"
    else:
        return "Referencia"

df_long["Situacion"] = df_long["Archivo"].apply(extraer_situacion)
# =========================================================
# ===== GRÁFICOS DE BARRAS POR ELECTRODO =====
# =========================================================

banda_objetivo = "Alpha"   # o Delta / Theta / Beta

for canal in canales:

    datos = df_long[df_long["Canal"] == canal]

    resumen = (
        datos
        .groupby("Situacion")[banda_objetivo]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        resumen["Situacion"],
        resumen["mean"],
        yerr=resumen["std"],
        capsize=5,
        alpha=0.8
    )

    plt.ylabel(f"Energía relativa {banda_objetivo} [%]")
    plt.xlabel("Situación")
    plt.title(f"{banda_objetivo} – Electrodo {canal}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
banda_objetivo = "Alpha"   # o Delta, Theta, Beta

media_por_electrodo = (
    df_long
    .groupby(["Canal", "Situacion"])[banda_objetivo]
    .mean()
    .reset_index()
)
media_global = (
    media_por_electrodo
    .groupby("Situacion")[banda_objetivo]
    .mean()
    .reset_index()
)
plt.figure(figsize=(7, 5))

plt.bar(
    media_global["Situacion"],
    media_global[banda_objetivo],
    alpha=0.8
)

plt.ylabel(f"Energía relativa {banda_objetivo} [%]")
plt.xlabel("Situación")
plt.title(f"{banda_objetivo} – Promedio global (4 electrodos)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

