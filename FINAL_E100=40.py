
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

def psd_welch(señal, fs=200):

    f, psd = sig.welch(
        señal,
        fs=fs,
        window='hann',
        nperseg=256,
        noverlap=128,
        scaling='density'
    )

    return f, psd

    return f, psd
def procesar_señal(eeg, fs, canales, titulo):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, canal in enumerate(canales):

        señal = eeg[:, i]
        señal = señal / np.std(señal)

        ff_half, psd_half = psd_welch(señal, fs)

        energia_total = np.sum(psd_half)
        energia_acum = np.cumsum(psd_half) / energia_total

        f95 = ff_half[np.where(energia_acum >= 0.95)[0][0]]
        f98 = ff_half[np.where(energia_acum >= 0.98)[0][0]]

        axs[i].plot(ff_half, 10*np.log10(psd_half + 1e-12), color="black")

        axs[i].axvline(f95, color='blue', linestyle='--', label=f'95%: {f95:.2f} Hz')
        axs[i].axvline(f98, color='purple', linestyle='--', label=f'98%: {f98:.2f} Hz')

        axs[i].set_title(f"Electrodo {canal}")
        axs[i].set_xlabel("Frecuencia [Hz]")
        axs[i].set_ylabel("PSD [dB]")
        axs[i].grid()
        axs[i].legend()

    fig.suptitle(titulo)

    plt.tight_layout()
    plt.show()
  
def procesar_bandas_con_plot(eeg, fs, canales, titulo):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    resultados = {}

    for i, canal in enumerate(canales):

        señal = eeg[:, i]
        señal = señal / np.std(señal)

        ff_half, psd_half = psd_welch(señal, fs)

        # limitar análisis espectral a 0–40 Hz
        fmax_eeg = 40
        mask = ff_half <= fmax_eeg
        ff_half = ff_half[mask]
        psd_half = psd_half[mask]

        energia_total = np.sum(psd_half)

        energias = {}

        axs[i].plot(ff_half, 10*np.log10(psd_half + 1e-12), color="black")

        for banda, (fmin, fmax) in bandas.items():

            idx = (ff_half >= fmin) & (ff_half < fmax)

            energias[banda] = np.sum(psd_half[idx]) / energia_total * 100

            axs[i].axvspan(
                fmin,
                fmax,
                color=colores_bandas[banda],
                alpha=0.25,
                label=banda
            )

        resultados[canal] = energias

        # eliminar duplicados en leyenda
        handles, labels = axs[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[i].legend(by_label.values(), by_label.keys())

        axs[i].set_title(f"Electrodo {canal}")
        axs[i].set_xlabel("Frecuencia [Hz]")
        axs[i].set_ylabel("PSD [dB]")
        axs[i].grid()

    fig.suptitle(titulo)

    plt.tight_layout()
    plt.show()

    return resultados
def energia_bandas_sin_plot(señal, fs):

    señal = señal / np.std(señal)

    ff_half, psd_half = psd_welch(señal, fs)

    # limitar análisis espectral a 0–40 Hz
    fmax_eeg = 40
    mask = ff_half <= fmax_eeg
    ff_half = ff_half[mask]
    psd_half = psd_half[mask]

    # energía total SOLO en 0–40 Hz
    energia_total = np.sum(psd_half)

    return {
        banda: np.sum(psd_half[(ff_half >= fmin) & (ff_half < fmax)]) / energia_total * 100
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
procesar_señal(
    eeg,
    fs,
    canales,
    "PSD EEG sin filtrar – P1"
)

procesar_señal(
    eeg_filtrado,
    fs,
    canales,
    "PSD EEG filtrada BP + NOTCH – P1"
)

# Energía por bandas (con plots)
energias = energia_bandas_sin_plot(
    eeg_filtrado,
    fs
)

# print(energias)

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
# BOXPLOTS referencia
situacion_objetivo = "Referencia"
canales = ["Cz", "T7", "F8", "P4"]
bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, canal in enumerate(canales):

    datos = df_long[
        (df_long["Situacion"] == situacion_objetivo) &
        (df_long["Canal"] == canal)
    ]

    data = [datos[banda].values for banda in bandas_lista]

    axs[i].boxplot(data, showfliers=True)
    axs[i].set_xticks(range(1, len(bandas_lista) + 1))
    axs[i].set_xticklabels(bandas_lista)
    axs[i].set_ylabel("Energía relativa [%]")
    axs[i].set_ylim(0, 80)
    axs[i].set_xlabel("Bandas EEG")
    axs[i].set_title(f"Electrodo {canal}")
    axs[i].grid(axis="y", alpha=0.3)

fig.suptitle(f"{situacion_objetivo}", fontsize=14)
plt.tight_layout()
plt.show()
# %% 
# BOXPLOTS Música Nativa - 4 electrodos en una sola figura

situacion_objetivo = "Música nativa"
canales = ["Cz", "T7", "F8", "P4"]
bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, canal in enumerate(canales):

    datos = df_long[
        (df_long["Situacion"] == situacion_objetivo) &
        (df_long["Canal"] == canal)
    ]

    data = [datos[banda].values for banda in bandas_lista]

    axs[i].boxplot(data, showfliers=True)
    axs[i].set_xticks(range(1, len(bandas_lista) + 1))
    axs[i].set_xticklabels(bandas_lista)
    axs[i].set_ylabel("Energía relativa [%]")
    axs[i].set_ylim(0, 80)
    axs[i].set_xlabel("Bandas EEG")
    axs[i].set_title(f"Electrodo {canal}")
    axs[i].grid(axis="y", alpha=0.3)

fig.suptitle(f"{situacion_objetivo}", fontsize=14)
plt.tight_layout()
plt.show()

# =========================================================
# # ===== BOXPLOT PROMEDIO =====
# # =========================================================

# situacion_objetivo = "Música nativa"   # cambiar

# bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# # Promedio entre electrodos por sujeto
# promedio_global = (
#     df_long[df_long["Situacion"] == situacion_objetivo]
#     .groupby("Persona")[bandas_lista]
#     .mean()
#     .reset_index()
# )

# data = [promedio_global[banda].values for banda in bandas_lista]

# plt.figure(figsize=(7, 5))
# plt.boxplot(data, showfliers=True)

# plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
# plt.ylabel("Energía relativa [%]")
# plt.xlabel("Bandas EEG")
# plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()


# %%
# BOXPLOTS Música extranjera
# BOXPLOTS Música Nativa - 4 electrodos en una sola figura

situacion_objetivo = "Música extranjera"
canales = ["Cz", "T7", "F8", "P4"]
bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, canal in enumerate(canales):

    datos = df_long[
        (df_long["Situacion"] == situacion_objetivo) &
        (df_long["Canal"] == canal)
    ]

    data = [datos[banda].values for banda in bandas_lista]

    axs[i].boxplot(data, showfliers=True)
    axs[i].set_xticks(range(1, len(bandas_lista) + 1))
    axs[i].set_xticklabels(bandas_lista)
    axs[i].set_ylabel("Energía relativa [%]")
    axs[i].set_ylim(0, 80)
    axs[i].set_xlabel("Bandas EEG")
    axs[i].set_title(f"Electrodo {canal}")
    axs[i].grid(axis="y", alpha=0.3)

fig.suptitle(f"{situacion_objetivo}", fontsize=14)
plt.tight_layout()
plt.show()

# # =========================================================
# # ===== BOXPLOT PROMEDIO =====
# # =========================================================

# situacion_objetivo = "Música extranjera"   # cambiar

# bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# # Promedio entre electrodos por sujeto
# promedio_global = (
#     df_long[df_long["Situacion"] == situacion_objetivo]
#     .groupby("Persona")[bandas_lista]
#     .mean()
#     .reset_index()
# )

# data = [promedio_global[banda].values for banda in bandas_lista]

# plt.figure(figsize=(7, 5))
# plt.boxplot(data, showfliers=True)

# plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
# plt.ylabel("Energía relativa [%]")
# plt.xlabel("Bandas EEG")
# plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()
# %%

# BOXPLOTS Música Neutral
# BOXPLOTS Música Nativa - 4 electrodos en una sola figura

situacion_objetivo = "Música neutra"
canales = ["Cz", "T7", "F8", "P4"]
bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, canal in enumerate(canales):

    datos = df_long[
        (df_long["Situacion"] == situacion_objetivo) &
        (df_long["Canal"] == canal)
    ]

    data = [datos[banda].values for banda in bandas_lista]

    axs[i].boxplot(data, showfliers=True)
    axs[i].set_xticks(range(1, len(bandas_lista) + 1))
    axs[i].set_xticklabels(bandas_lista)
    axs[i].set_ylabel("Energía relativa [%]")
    axs[i].set_ylim(0, 80)
    axs[i].set_xlabel("Bandas EEG")
    axs[i].set_title(f"Electrodo {canal}")
    axs[i].grid(axis="y", alpha=0.3)

fig.suptitle(f"{situacion_objetivo}", fontsize=14)
plt.tight_layout()
plt.show()

# # =========================================================
# # ===== BOXPLOT PROMEDIO =====
# # =========================================================

# situacion_objetivo = "Música neutra"   # cambiar

# bandas_lista = ["Delta", "Theta", "Alpha", "Beta"]

# # Promedio entre electrodos por sujeto
# promedio_global = (
#     df_long[df_long["Situacion"] == situacion_objetivo]
#     .groupby("Persona")[bandas_lista]
#     .mean()
#     .reset_index()
# )

# data = [promedio_global[banda].values for banda in bandas_lista]

# plt.figure(figsize=(7, 5))
# plt.boxplot(data, showfliers=True)

# plt.xticks(range(1, len(bandas_lista) + 1), bandas_lista)
# plt.ylabel("Energía relativa [%]")
# plt.xlabel("Bandas EEG")
# plt.title(f"{situacion_objetivo} – Promedio global (4 electrodos)")
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()
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

# =========================================================
# COMPARACIÓN ENTRE CINCO PERSONAS EN TODAS LAS SITUACIONES
# =========================================================

persona1 = "P2"
persona2 = "P3"
persona3 = "P5"
persona4 = "P7"
persona5 = "P9"

personas_comparar = [persona1, persona2, persona3, persona4, persona5]

situaciones = [
    "Referencia",
    "Música nativa",
    "Música extranjera",
    "Música neutra"
]

for situacion_obj in situaciones:

    fig, axs = plt.subplots(5, 4, figsize=(16,12), constrained_layout=True)

    for fila, persona in enumerate(personas_comparar):

        # buscar archivo con esa situación
        archivo_obj = None

        for archivo in personas[persona]:
            if extraer_situacion(archivo) == situacion_obj:
                archivo_obj = archivo
                break

        if archivo_obj is None:
            continue

        eeg, _ = cargar_eeg(f"{base_path}\\{archivo_obj}", fs)

        # =========================
        # CONVERTIR µV → V
        # =========================
        eeg = eeg * 1e-6

        # =========================
        # FILTRADO
        # =========================

        eeg_filtrado = np.zeros_like(eeg)

        for i in range(4):
            x = sig.sosfiltfilt(sos_notch, eeg[:, i])
            eeg_filtrado[:, i] = sig.sosfiltfilt(sos, x)

        # =========================
        # ESPECTROGRAMAS
        # =========================

        for i, canal in enumerate(canales):

            f, t, Sxx = sig.spectrogram(
                eeg_filtrado[:, i],
                fs=fs,
                nperseg=fs*2,
                noverlap=fs
            )

            im = axs[fila, i].pcolormesh(
                t,
                f,
                10*np.log10(Sxx + 1e-12),
                shading='gouraud'
            )

            axs[fila, i].set_title(canal)
            axs[fila, i].set_ylim(0,40)

            if i == 0:
                axs[fila, i].set_ylabel(f"{persona}\nFrecuencia [Hz]")

            axs[fila, i].set_xlabel("Tiempo [s]")

    fig.colorbar(im, ax=axs, shrink=0.7, label="Potencia [dB]")

    fig.suptitle(f"Comparación espectrogramas – {situacion_obj}", fontsize=14)

    plt.show()
# %%
import mne
import numpy as np
import matplotlib.pyplot as plt

def topomap_bandas(df_long, situacion):

    bandas = ["Delta", "Theta", "Alpha", "Beta"]
    canales = ["Cz", "T7", "F8", "P4"]

    # posiciones electrodos (orden debe coincidir con canales)
    pos = np.array([
        [0, 0.3],     # Cz
        [-0.5, 0],    # T7
        [0.4, 0.6],   # F8
        [0.4, -0.5]   # P4
    ])

    datos = (
        df_long[df_long["Situacion"] == situacion]
        .groupby("Canal")[bandas]
        .mean()
        .loc[canales]
    )

    fig, axs = plt.subplots(1, 4, figsize=(12,3))

    for i, banda in enumerate(bandas):

        valores = datos[banda].values

    im, _ = mne.viz.plot_topomap(
    valores,
    pos,
    axes=axs[i],
    show=False,
    cmap="viridis",
    extrapolate="head"
)


    axs[i].set_title(banda)

        

    fig.suptitle(situacion)

    cbar = fig.colorbar(im, ax=axs, shrink=0.7)
    cbar.set_label("Energía relativa [%]")

    plt.show()
topomap_bandas(df_long, "Referencia")
topomap_bandas(df_long, "Música nativa")
topomap_bandas(df_long, "Música extranjera")
topomap_bandas(df_long, "Música neutra")
# %%

electrodo = "Cz"

situaciones = ["Referencia",
               "Música nativa",
               "Música extranjera",
               "Música neutra"]

bandas = ["Delta", "Theta", "Alpha", "Beta"]

# color por banda
colores_bandas = {
    "Delta": "tab:blue",
    "Theta": "tab:orange",
    "Alpha": "tab:green",
    "Beta": "tab:red"
}

data = []
labels = []
colores = []

for banda in bandas:

    for situacion in situaciones:

        datos = df_long[
            (df_long["Canal"] == electrodo) &
            (df_long["Situacion"] == situacion)
        ]

        data.append(datos[banda].values)
        labels.append(f"{banda}\n{situacion}")

        colores.append(colores_bandas[banda])


plt.figure(figsize=(12,5))

bp = plt.boxplot(
    data,
    showfliers=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2)
)

# colorear cajas
for patch, color in zip(bp["boxes"], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

plt.xticks(range(1, len(labels)+1), labels, rotation=45)

plt.ylabel("Energía relativa [%]")
plt.ylim(0,80)

plt.title(f"Electrodo {electrodo}")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
# %%

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

canales = ["Cz", "T7", "F8", "P4"]

situaciones = ["Referencia",
               "Música nativa",
               "Música extranjera",
               "Música neutra"]

# abreviaturas para la figura
situaciones_abrev = ["REF", "MN", "ME", "NEU"]

bandas = ["Delta", "Theta", "Alpha", "Beta"]

# colores por banda
colores_bandas = {
    "Delta": "tab:blue",
    "Theta": "tab:orange",
    "Alpha": "tab:green",
    "Beta": "tab:red"
}

fig, axs = plt.subplots(4,1, figsize=(10,12), sharey=True)

for j, canal in enumerate(canales):

    ax = axs[j]

    data = []
    labels = []
    colores = []

    for banda in bandas:

        for i, situacion in enumerate(situaciones):

            datos = df_long[
                (df_long["Canal"] == canal) &
                (df_long["Situacion"] == situacion)
            ]

            data.append(datos[banda].values)
            labels.append(situaciones_abrev[i])

            colores.append(colores_bandas[banda])

    bp = ax.boxplot(
        data,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2)
    )

    # colorear cajas
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylim(0,80)

    # solo el último gráfico muestra etiquetas X
    if j == len(canales)-1:
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=0)
    else:
        ax.set_xticks([])

    ax.set_title(f"({chr(97+j)})", fontsize=11)

    ax.set_ylabel("Energía [%]")

    ax.grid(axis="y", alpha=0.3)

# -------- leyenda de bandas --------

legend_handles = [
    mpatches.Patch(color="tab:blue", label="Delta"),
    mpatches.Patch(color="tab:orange", label="Theta"),
    mpatches.Patch(color="tab:green", label="Alpha"),
    mpatches.Patch(color="tab:red", label="Beta")
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

fig.suptitle("Distribución de energía relativa por banda EEG", fontsize=14)

plt.tight_layout(rect=[0,0.05,1,0.96])

plt.show()
# %%

