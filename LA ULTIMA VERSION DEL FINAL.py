
"""
Created on Mon Feb  2 22:25:08 2026

@author: Vale
"""

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


def cargar_eeg(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)
    eeg = datos.iloc[9000:33000, 1:5].apply(pd.to_numeric).values

    # quitar offset DC por canal
    eeg -= np.mean(eeg, axis=0)

    tiempo = np.arange(len(eeg)) / fs
    return eeg, tiempo
    
def cargar_eeg_filtrado(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    # columnas EEG originales del archivo
    eeg_raw = datos.iloc[:, 1:5].apply(pd.to_numeric)

    # ORDEN REAL del archivo:
    # archivo trae: [p4, cz, F8, T7]
    # queremos:     [T7, F8, Cz, P4]
    orden_columnas = [3, 2, 1, 0]

    eeg = eeg_raw.iloc[:, orden_columnas].values

    # quitar offset DC por canal
    eeg -= np.mean(eeg, axis=0)

    tiempo = np.arange(len(eeg)) / fs
    return eeg, tiempo

def espectrograma_eeg(señal, fs, canal, situacion):

    f, t, Sxx = sig.spectrogram(
        señal,
        fs=fs,
        window='hann',
        nperseg=400,
        noverlap=200,
        scaling='density'
    )

    plt.figure(figsize=(7,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
    plt.ylim(0,40)

    plt.ylabel("Frecuencia [Hz]")
    plt.xlabel("Tiempo [s]")
    plt.title(f"Espectrograma – {canal} – {situacion}")
    plt.colorbar(label="PSD [dB]")

    plt.tight_layout()
    plt.show()
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
# NOTCH 50 Hz – ESTABLE PARA EEG
# =========================================================

f0 = 50.0      # frecuencia de red
Q = 30.0       # factor de calidad (25–35 recomendado)

b_notch, a_notch = sig.iirnotch(f0, Q, fs=fs)
sos_notch = sig.tf2sos(b_notch, a_notch)



# =========================================================
# ARCHIVOS
# =========================================================

base_path = r"C:\Users\Vale\Documents\APS\APS_vale\TPS"

personas = {
    "P1": ["s01_ex01_s01.txt", "s01_ex05.txt", "s01_ex06.txt", "s01_ex07.txt"],
    "P2": ["s02_ex01_s01.txt", "s02_ex05.txt", "s02_ex06.txt", "s02_ex07.txt"],
   "P3": ["s03_ex01_s01.txt", "s03_ex05.txt", "s03_ex06.txt", "s03_ex07.txt"],
    "P4": ["s04_ex01_s01.txt", "s04_ex05.txt", "s04_ex06.txt", "s04_ex07.txt"],
   "P5": ["s05_ex01_s01.txt", "s05_ex05.txt", "s05_ex06.txt", "s05_ex07.txt"],
    "P6": ["s06_ex01_s01.txt", "s06_ex05.txt", "s06_ex06.txt", "s06_ex07.txt"],
    "P7": ["s07_ex01_s01.txt", "s07_ex05.txt", "s07_ex06.txt", "s07_ex07.txt"],
   "P8": ["s08_ex01_s01.txt", "s08_ex05.txt", "s08_ex06.txt", "s08_ex07.txt"],
    "P9": ["s09_ex01_s01.txt", "s09_ex05.txt", "s09_ex06.txt", "s09_ex07.txt"],
   "P10": ["s10_ex01_s01.txt", "s10_ex05.txt", "s10_ex06.txt", "s10_ex07.txt"],
    "P1filtrado": ["s01_ex01_s01.csv"],
}

# %%
# =========================================================

# ===== PRIMERA PERSONA – ANÁLISIS COMPLETO =====
# =========================================================

persona_ref = "P1"
archivo_ref = personas[persona_ref][0]

eeg, tiempo = cargar_eeg(f"{base_path}\\{archivo_ref}", fs)

persona_ref_2 = "P1filtrado"
archivo_ref_2 = personas[persona_ref_2][0]

eeg_2, tiempo_2 = cargar_eeg_filtrado(f"{base_path}\\{archivo_ref_2}", fs)
# Gráfica temporal
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for i in range(4):
    ax[i].plot(tiempo, eeg[:, i])
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].grid(True)
ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()

# =========================================================
# FILTRADO EEG: NOTCH + PASABANDA
# =========================================================
eeg_filtrado = np.zeros_like(eeg)

for i in range(4):
    x = sig.sosfiltfilt(sos_notch, eeg[:, i])   # NOTCH
    eeg_filtrado[:, i] = sig.sosfiltfilt(sos, x)  # PASABANDA


# PSD + ancho de banda
for i, canal in enumerate(canales):
    procesar_señal(eeg[:, i], fs, f"PSD EEG – {canal}")
    procesar_señal(eeg_filtrado[:, i], fs, f"PSD EEG FILTRADA BP + NOTCH – {canal}")
    
# Energía por bandas (con plots)
for i, canal in enumerate(canales):
    procesar_bandas_con_plot(
        eeg_filtrado[:, i],
        fs,
        f"Energía por bandas – {canal}"
    )

# Gráfica temporal
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
for i in range(4): 
    ax[i].plot(tiempo, eeg_filtrado[:, i])
    ax[i].plot(tiempo_2, eeg_2[:, i])
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].grid(True)
ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()
# %%

# Gráfica temporal
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

for i in range(4):
    ax[i].plot(
        tiempo[23000:23100],
        eeg_filtrado[23000:23100, i],
        color= "pink",
        label="Filtrado por mi"
    )
    ax[i].plot(
        tiempo_2[23000:23100],
        eeg_2[23000:23100, i],
        color= "purple",
        label="Referencia"
    )
    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].grid(True)
    ax[i].legend()

ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()
# %%
 #Aca para que se vean que estan en fase las señales lo que hago es sumarle una "demora"
 #En realidad no eso lo que pasa, porque se supone que si usamos filtfilt la demora esta contrarrestada
 #Aunque la verdad no me acuerdo muy bien
 # Entonces creo que se trata de que capaz esta corrida en tiempo la señal simplemente eso 
demora = 1

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8),)

for i in range(4):

    ax[i].plot(
        tiempo[23000:23100 - demora],
        eeg_filtrado[23000 + demora:23100, i],
        color= "pink",
        label="Filtrado por mí (demora contrarrestada)"
    
    )

    ax[i].plot(
        tiempo_2[23000:23100 - demora],
        eeg_2[23000:23100 - demora, i],
        color= "purple",
        label="Referencia"
    )

    ax[i].set_title(f"Canal {canales[i]}")
    ax[i].grid(True)
    ax[i].legend()

ax[-1].set_xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()
# %%

# =========================================================
# SEGMENTACIÓN EN VENTANAS DE 2 s
# =========================================================

duracion_seg = 2
Nwin = duracion_seg * fs   # 400 muestras

energias_2s = []

for inicio in range(0, len(eeg_filtrado) - Nwin, Nwin):
    segmento = eeg_filtrado[inicio:inicio + Nwin, :]

    energias_canales = {}
    for i, canal in enumerate(canales):
        energias_canales[canal] = energia_bandas_sin_plot(
            segmento[:, i],
            fs
        )

    energias_2s.append(energias_canales)
# %%

# =========================================================
# ===== RESTO DE PERSONAS – PROCESAMIENTO CON SEGMENTACIÓN
# =========================================================

duracion_seg = 2
Nwin = duracion_seg * fs   # 400 muestras

resultados = {}

for persona, archivos in personas.items():
    for archivo in archivos:

        if persona == persona_ref and archivo == archivo_ref:
            continue

        eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)

        # ---------------------------------------------
        # FILTRADO: NOTCH + PASABANDA (cero fase)
        # ---------------------------------------------
        eeg_filtrado = np.zeros_like(eeg)

        for i in range(4):
            x = sig.sosfiltfilt(sos_notch, eeg[:, i])   # NOTCH 50 Hz
            eeg_filtrado[:, i] = sig.sosfiltfilt(sos, x)  # PASABANDA

        # ---------------------------------------------
        # SEGMENTACIÓN + ENERGÍA
        # ---------------------------------------------
        resultados_archivo = {}

        for i, canal in enumerate(canales):

            energias_ventanas = []

            for inicio in range(0, len(eeg_filtrado) - Nwin, Nwin):
                segmento = eeg_filtrado[inicio:inicio + Nwin, i]

                energias_ventanas.append(
                    energia_bandas_sin_plot(segmento, fs)
                )

            # promedio de ventanas
            resultados_archivo[canal] = (
                pd.DataFrame(energias_ventanas).mean().to_dict()
            )

        resultados[(persona, archivo)] = resultados_archivo

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
        return "Música nativa"
    elif "ex06" in nombre:
        return "Música extranjera"
    elif "ex07" in nombre:
        return "Música neutra"
    else:
        return "Referencia"

df_long["Situacion"] = df_long["Archivo"].apply(extraer_situacion)
# %%
# BOXPLOTS Música Nativa

# =========================================================
# ===== BOXPLOT: Música Nativa Canal Cz =====
# =========================================================

situacion_objetivo = "Música nativa"   # cambiar según corresponda
canal_objetivo = "Cz"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal T7 =====
# =========================================================

situacion_objetivo = "Música nativa"   # cambiar según corresponda
canal_objetivo = "T7"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal F8 =====
# =========================================================

situacion_objetivo = "Música nativa"   # cambiar según corresponda
canal_objetivo = "F8"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# =========================================================
# ===== BOXPLOT: Música Nativa Canal P4 =====
# =========================================================

situacion_objetivo = "Música nativa"   # cambiar según corresponda
canal_objetivo = "P4"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT PROMEDIO =====
# =========================================================

situacion_objetivo = "Música nativa"   # cambiar

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# Promedio entre electrodos por sujeto
promedio_global = (
    df_long[df_long["Situacion"] == situacion_objetivo]
    .groupby("Persona")[bandas_lista]
    .mean()
    .reset_index()
)

data = [promedio_global[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# BOXPLOTS Música extranjera

# =========================================================
# ===== BOXPLOT: Música extranjera Canal Cz =====
# =========================================================

situacion_objetivo = "Música extranjera"   # cambiar según corresponda
canal_objetivo = "Cz"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal T7 =====
# =========================================================

situacion_objetivo = "Música extranjera"   # cambiar según corresponda
canal_objetivo = "T7"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal F8 =====
# =========================================================

situacion_objetivo = "Música extranjera"   # cambiar según corresponda
canal_objetivo = "F8"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# =========================================================
# ===== BOXPLOT: Música Nativa Canal P4 =====
# =========================================================

situacion_objetivo = "Música extranjera"   # cambiar según corresponda
canal_objetivo = "P4"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT PROMEDIO =====
# =========================================================

situacion_objetivo = "Música extranjera"   # cambiar

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# Promedio entre electrodos por sujeto
promedio_global = (
    df_long[df_long["Situacion"] == situacion_objetivo]
    .groupby("Persona")[bandas_lista]
    .mean()
    .reset_index()
)

data = [promedio_global[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# %%

# BOXPLOTS Música C

# =========================================================
# ===== BOXPLOT: Música neutra Canal Cz =====
# =========================================================

situacion_objetivo = "Música neutra"   # cambiar según corresponda
canal_objetivo = "Cz"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal T7 =====
# =========================================================

situacion_objetivo = "Música neutra"   # cambiar según corresponda
canal_objetivo = "T7"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT: Música Nativa Canal F8 =====
# =========================================================

situacion_objetivo = "Música neutra"   # cambiar según corresponda
canal_objetivo = "F8"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# =========================================================
# ===== BOXPLOT: Música Nativa Canal P4 =====
# =========================================================

situacion_objetivo = "Música neutra"    # cambiar según corresponda
canal_objetivo = "P4"                # T7, F8, Cz, P4

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

datos = df_long[
    (df_long["Situacion"] == situacion_objetivo) &
    (df_long["Canal"] == canal_objetivo)
]

data = [datos[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Electrodo {canal_objetivo}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ===== BOXPLOT PROMEDIO =====
# =========================================================

situacion_objetivo = "Música neutra"   # cambiar

bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# Promedio entre electrodos por sujeto
promedio_global = (
    df_long[df_long["Situacion"] == situacion_objetivo]
    .groupby("Persona")[bandas_lista]
    .mean()
    .reset_index()
)

data = [promedio_global[banda].values for banda in bandas_lista]

plt.figure(figsize=(7, 5))
plt.boxplot(data, showfliers=True)

plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
plt.ylabel("Energía relativa [%]")
plt.xlabel("Bandas EEG")
plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# %%

# =========================================================
# SEGMENTACIÓN EN VENTANAS DE 4 s
# =========================================================

# duracion_seg = 4
# Nwin = duracion_seg * fs   # 800 muestras

# energias_4s = []

# for inicio in range(0, len(eeg_filtrado) - Nwin, Nwin):
#     segmento = eeg_filtrado[inicio:inicio + Nwin, :]

#     energias_canales = {}
#     for i, canal in enumerate(canales):
#         energias_canales[canal] = energia_bandas_sin_plot(
#             segmento[:, i],
#             fs
#         )

#     energias_4s.append(energias_canales)
    
# %%

# %%
# =========================================================
# MATRICES DE CORRELACIÓN ENTRE ELECTRODOS POR SITUACIÓN
# =========================================================

correlaciones_situaciones = {
    "Reposo": [],
    "Música nativa": [],
    "Música extranjera": [],
    "Música neutra": []
}

for persona, archivos in personas.items():

    for archivo in archivos:

        # cargar señal
        eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)

        # ---------------------------------------------
        # FILTRADO: NOTCH + PASABANDA
        # ---------------------------------------------
        eeg_filtrado = np.zeros_like(eeg)

        for i in range(4):
            x = sig.sosfiltfilt(sos_notch, eeg[:, i])
            eeg_filtrado[:, i] = sig.sosfiltfilt(sos, x)

        # ---------------------------------------------
        # CALCULAR MATRIZ DE CORRELACIÓN
        # ---------------------------------------------
        matriz_corr = np.corrcoef(eeg_filtrado.T)

        # ---------------------------------------------
        # IDENTIFICAR SITUACIÓN
        # ---------------------------------------------
        situacion = extraer_situacion(archivo)

        if situacion == "Referencia":
            correlaciones_situaciones["Reposo"].append(matriz_corr)

        elif situacion == "Música nativa":
            correlaciones_situaciones["Música nativa"].append(matriz_corr)

        elif situacion == "Música extranjera":
            correlaciones_situaciones["Música extranjera"].append(matriz_corr)

        elif situacion == "Música neutra":
            correlaciones_situaciones["Música neutra"].append(matriz_corr)


# =========================================================
# PROMEDIAR MATRICES POR SITUACIÓN
# =========================================================

for situacion, matrices in correlaciones_situaciones.items():

    if len(matrices) == 0:
        continue

    matriz_prom = np.mean(matrices, axis=0)

    matriz_df = pd.DataFrame(
        matriz_prom,
        index=canales,
        columns=canales
    )

    print("\n====================================")
    print(f"Matriz de correlación – {situacion}")
    print("====================================")
    print(matriz_df)
# %%
# %%
# =========================================================
# ESPECTROGRAMAS POR SITUACIÓN Y ELECTRODO (P1)
# =========================================================

persona_espectro = "P1"

for archivo in personas[persona_espectro]:

    eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)

    # FILTRADO
    eeg_filtrado = np.zeros_like(eeg)

    for i in range(4):
        x = sig.sosfiltfilt(sos_notch, eeg[:, i])
        eeg_filtrado[:, i] = sig.sosfiltfilt(sos, x)

    situacion = extraer_situacion(archivo)

    for i, canal in enumerate(canales):

        espectrograma_eeg(
            eeg_filtrado[:, i],
            fs,
            canal,
            situacion
        )
