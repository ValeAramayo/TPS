import os
import numpy as np
import pandas as pd # Util para aramar las tablas
import matplotlib.pyplot as plt #Necesaria para los graficos
import matplotlib.patches as mpatches # La usamos para armar el boxplot
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
# %% DEFINICIONES
bandas = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30)
}

colores_bandas = {
    "Delta": "#1f77b4",
    "Theta": "#2ca02c",
    "Alpha": "#ff7f0e",
    "Beta": "#d62728"
}

legend_handles = [
    mpatches.Patch(color="#1f77b4", label="Delta"),
    mpatches.Patch(color="#2ca02c", label="Theta"),
    mpatches.Patch(color="#ff7f0e", label="Alpha"),
    mpatches.Patch(color="#d62728", label="Beta")
]
canales = ["T7", "F8", "Cz", "P4"]

colores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"] #correspondiente a cada canal


situaciones = ["Referencia",
               "Música nativa",
               "Música extranjera",
               "Música neutra"]

# abreviaturas para la figura
situaciones_abrev = ["REF", "MN", "ME", "NEU"]

# %% VENTANAS POR ARCHIVO


ventanas = {
    # ===== PERSONA 1 =====
    "s01_ex02_s01.txt": (3000, 27000),
    "s01_ex05.txt": (13200, 37200),
    "s01_ex06.txt": (13200, 37200),
    "s01_ex07.txt": (1600, 25600),
    "s01_ex02_s01.csv": (3000, 27000),

    # ===== PERSONA 2 =====
    "s02_ex02_s01.txt": (2000 , 26000),
    "s02_ex05.txt": (11000 , 35000),
    "s02_ex06.txt": (12000 , 36000),
    "s02_ex07.txt": (6000 , 30000),
    

    # ===== PERSONA 3 =====
    "s03_ex02_s01.txt": (4800, 28800),
    "s03_ex05.txt": (6000 , 30000),
    "s03_ex06.txt": (16200 , 40200),
    "s03_ex07.txt": (6000 , 30000),
    

    # ===== PERSONA 4 =====
    "s04_ex02_s01.txt": (2000 , 26000),
    "s04_ex05.txt": (2000 , 26000),
    "s04_ex06.txt": (2000 , 26000),
    "s04_ex07.txt": (2000 , 26000),
    

    # ===== PERSONA 5 =====
    "s05_ex02_s01.txt": (2000 , 26000),
    "s05_ex05.txt": (6000 , 30000),
    "s05_ex06.txt": (2000 , 26000),
    "s05_ex07.txt": (2000 , 26000),
    

    # ===== PERSONA 6 =====
    "s06_ex02_s01.txt": (2000 , 26000),
    "s06_ex05.txt": (2000 , 26000),
    "s06_ex06.txt": (2000 , 26000),
    "s06_ex07.txt": (2000 , 26000),
    

    # ===== PERSONA 7 =====
    "s07_ex02_s01.txt": (3000 , 27000),
    "s07_ex05.txt": (2000 , 26000),
    "s07_ex06.txt": (9000 , 33000),
    "s07_ex07.txt": (2000 , 26000),
    

    # ===== PERSONA 8 =====
    "s08_ex02_s01.txt": (2000 , 26000),
    "s08_ex05.txt": (2000 , 26000),
    "s08_ex06.txt": (2000 , 26000),
    "s08_ex07.txt": (2000 , 26000),
  

    # ===== PERSONA 9 =====
    "s09_ex02_s01.txt": (2000 , 26000),
    "s09_ex05.txt": (1600, 25600),
    "s09_ex06.txt": (2000 , 26000),
    "s09_ex07.txt": (2000 , 26000),
    

    # ===== PERSONA 10 =====
    "s10_ex02_s01.txt": (2000 , 26000),
    "s10_ex05.txt": (2000 , 26000),
    "s10_ex06.txt": (2000 , 26000),
    "s10_ex07.txt": ( 8000, 32000),
  
}

# %% FUNCIONES 

# =========================
# CARGAR EL EEG POR EXPERIMENTO
# =========================
def cargar_eeg(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    nombre_archivo = os.path.basename(path)

    if nombre_archivo not in ventanas:
        raise ValueError(f"No definiste ventana para {nombre_archivo}")

    inicio, fin = ventanas[nombre_archivo]

    eeg = datos.iloc[inicio:fin, 1:5].apply(pd.to_numeric).values

    eeg -= np.mean(eeg, axis=0)
    tiempo = np.arange(len(eeg)) / fs

    return eeg, tiempo


def cargar_eeg_filtrado(path, fs):
    datos = pd.read_csv(path, sep=",", skiprows=5)

    nombre_archivo = os.path.basename(path)

    if nombre_archivo not in ventanas:
        raise ValueError(f"No definiste ventana para {nombre_archivo}")

    inicio, fin = ventanas[nombre_archivo]

    eeg_raw = datos.iloc[inicio:fin, 1:5].apply(pd.to_numeric)

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

    b, a = butter_bandpass(1, 40, fs, order=1)
    eeg_filtrado = filtfilt(b, a, eeg, axis=0)

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
# PSD POR SEGMENTO - Calculamos PSD con Método de Welch
# =========================

def psd_segmentos(segmentos, fs):
    psd_list = []

    for seg in segmentos:
        f, Pxx = welch(seg, fs=fs, nperseg=len(seg), axis=0)
        psd_list.append(Pxx)

    return f, np.array(psd_list)
# =========================
# Detección del 95% de la energía
# =========================
def analizar_psd_existente(f, psd, canales, titulo="", plot=True):

    resultados = []

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

    for i, canal in enumerate(canales):

        psd_ch = psd[:, i]

        energia_total = np.sum(psd_ch)
        energia_acum = np.cumsum(psd_ch) / energia_total

        f95 = f[np.where(energia_acum >= 0.95)[0][0]]
        f98 = f[np.where(energia_acum >= 0.98)[0][0]]

        resultados.append({
            "canal": canal,
            "f95": f95,
            "f98": f98
        })

        if plot:
            axs[i].plot(f, 10*np.log10(psd_ch + 1e-12), color="black")
            axs[i].axvline(f95, color='blue', linestyle='--', label=f'95%: {f95:.2f} Hz')
            axs[i].axvline(f98, color='purple', linestyle='--', label=f'98%: {f98:.2f} Hz')

            axs[i].set_title(f"({chr(97+i)})")
            axs[i].set_xlabel("Frecuencia [Hz]")
            axs[i].set_ylabel("PSD [dB]")
            axs[i].grid()
            axs[i].legend()

    if plot:
        fig.suptitle(titulo)
        plt.tight_layout()
        plt.show()

    return resultados
# =========================
# Detección de energia relativa por bandas con opcion de mostrar o no el grafico
# =========================
def analizar_bandas_psd(f, psd, canales, bandas, colores, colores_bandas, plot=True):

    resultados = {}

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

    for i, canal in enumerate(canales):

        psd_ch = psd[:, i]

        # limitar a 0–40 Hz
        mask = f <= 40
        f_lim = f[mask]
        psd_ch = psd_ch[mask]

        energia_total = np.sum(psd_ch)

        energias = {}

        if plot:
            axs[i].plot(f_lim, 10*np.log10(psd_ch + 1e-12), color=colores[i])

        for banda, (fmin, fmax) in bandas.items():

            idx = (f_lim >= fmin) & (f_lim < fmax)

            energia_banda = np.sum(psd_ch[idx])
            energias[banda] = (energia_banda / energia_total) * 100

            if plot:
                axs[i].axvspan(
                    fmin,
                    fmax,
                    color=colores_bandas[banda],
                    alpha=0.25,
                    label=banda
                )

        resultados[canal] = energias

        if plot:
            # eliminar duplicados en leyenda
            handles, labels = axs[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[i].legend(by_label.values(), by_label.keys())

            axs[i].set_title(f"({chr(97+i)})")
            axs[i].set_xlabel("Frecuencia [Hz]")
            axs[i].set_ylabel("PSD [dB]")
            axs[i].grid()

    if plot:
        plt.tight_layout()
        plt.show()

    return resultados

# =========================
# Procesamiento de la señal devuelve frecuencia, energia relativa y frecuencia
# =========================
def procesar_eeg_psd_media(eeg, fs, canales, bandas):

    # FILTRADO
    eeg_filtrado = aplicar_filtro(eeg, fs)
    # SEGMENTACIÓN (4 s)
    segmentos = segmentar(eeg_filtrado, fs, duracion=4)
    # PSD POR SEGMENTO
    f, psd_segs = psd_segmentos(segmentos, fs)
    # PSD MEDIA
    psd_media = np.mean(psd_segs, axis=0)
    # ENERGÍA RELATIVA POR BANDAS
    energia_relativa_bandas = {}

    for i, canal in enumerate(canales):

        psd_ch = psd_media[:, i]
        energia_total = np.sum(psd_ch)

        energias = {}

        for banda, (fmin, fmax) in bandas.items():
            idx = (f >= fmin) & (f < fmax)
            energia_banda = np.sum(psd_ch[idx])
            energias[banda] = (energia_banda / energia_total) * 100

        energia_relativa_bandas[canal] = energias

    return f, psd_media, energia_relativa_bandas

# =========================
# Funcion para el calculo de la conectividad por bandas
# =========================

def calcular_conectividad_banda(eeg_filtrados, fs, banda, canales, bandas):

    # --- obtener rango de la banda ---
    f_low, f_high = bandas[banda]

    # --- filtro ---
    def filtrar_banda(eeg, fs, f_low, f_high, order=1):
        nyq = fs / 2
        b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
        return filtfilt(b, a, eeg, axis=0)

    # --- estructura ---
    conectividad = {
        "Reposo": [],
        "Música nativa": [],
        "Música extranjera": [],
        "Música neutra": []
    }

    # --- cálculo ---
    for persona in eeg_filtrados:
        for situacion, eeg in eeg_filtrados[persona].items():

            eeg = eeg[:, [0, 1, 2, 3]]  # ordenar canales

            eeg_band = filtrar_banda(eeg, fs, f_low, f_high)

            matriz_corr = np.corrcoef(eeg_band.T)

            if situacion == "Referencia":
                conectividad["Reposo"].append(matriz_corr)
            elif situacion in conectividad:
                conectividad[situacion].append(matriz_corr)

    # --- gráfico ---
    fig, axs = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)
    situaciones_lista = list(conectividad.keys())

    for idx, situacion in enumerate(situaciones_lista):

        matrices = conectividad[situacion]
        if len(matrices) == 0:
            continue

        matriz_prom = np.mean(matrices, axis=0)

        ax = axs[idx//2, idx%2]
        im = ax.imshow(matriz_prom, vmin=-1, vmax=1)

        ax.set_title(situacion)
        ax.set_xticks(range(len(canales)))
        ax.set_yticks(range(len(canales)))
        ax.set_xticklabels(canales)
        ax.set_yticklabels(canales)

        for i in range(len(canales)):
            for j in range(len(canales)):
                valor = matriz_prom[i, j]

                ax.text(j, i, f"{valor:.2f}",
        ha='center', va='center',
        color='black', fontsize=8)

    # --- colorbar ---
    fig.colorbar(im, ax=axs, shrink=0.8)

    plt.suptitle(f"Conectividad - Banda {banda}")
    plt.show()
# %% LISTA DE PERSONAS Y ARCHIVOS
# =========================

personas = {
    "P1": ["s01_ex02_s01.txt", "s01_ex05.txt", "s01_ex06.txt", "s01_ex07.txt"],
    "P2": ["s02_ex02_s01.txt", "s02_ex05.txt", "s02_ex06.txt", "s02_ex07.txt"],
    "P3": ["s03_ex02_s01.txt", "s03_ex05.txt", "s03_ex06.txt", "s03_ex07.txt"],
    "P4": ["s04_ex02_s01.txt", "s04_ex05.txt", "s04_ex06.txt", "s04_ex07.txt"],
    "P5": ["s05_ex02_s01.txt", "s05_ex05.txt", "s05_ex06.txt", "s05_ex07.txt"],
    "P6": ["s06_ex02_s01.txt", "s06_ex05.txt", "s06_ex06.txt", "s06_ex07.txt"],
    "P7": ["s07_ex02_s01.txt", "s07_ex05.txt", "s07_ex06.txt", "s07_ex07.txt"],
    "P8": ["s08_ex02_s01.txt", "s08_ex05.txt", "s08_ex06.txt", "s08_ex07.txt"],
    "P9": ["s09_ex02_s01.txt", "s09_ex05.txt", "s09_ex06.txt", "s09_ex07.txt"],
    "P10": ["s10_ex02_s01.txt", "s10_ex05.txt", "s10_ex06.txt", "s10_ex07.txt"],
}

base_path = r"C:\Users\Vale\Documents\APS\APS_vale\TPS"
fs = 200

# =========================
# CARGA MASIVA y preprocesado
# =========================

resultados = {}

for persona, archivos in personas.items():
    resultados[persona] = {}

    for archivo in archivos:
        path = os.path.join(base_path, archivo)

        # 1. cargar con ventana (ya aplicada)
        eeg, _ = cargar_eeg(path, fs)

        # 2. filtrar
        eeg_filtrado = aplicar_filtro(eeg, fs)

        # 3. segmentar
        segmentos = segmentar(eeg_filtrado, fs)

        # 4. PSD por segmento
        f, psd_segs = psd_segmentos(segmentos, fs)

        # 5. PROMEDIO FINAL 
        psd_mean = np.mean(psd_segs, axis=0)

        # guardamos SOLO lo necesario
        resultados[persona][archivo] = {
            "frecuencias": f,
            "psd_mean": psd_mean
        }

print("PSD promedio calculada para todos")
# %% EJEMPLO PERSONA 1
fs = 200
path_crudo = "s01_ex02_s01.txt"
path_filtrado = "s01_ex02_s01.csv"# (Filtrado por la institución que realizó el experimento)

# CARGA
eeg_crudo, t = cargar_eeg(path_crudo, fs)
eeg_filtrado_archivo, _ = cargar_eeg_filtrado(path_filtrado, fs)

# FILTRADO PROPIO
eeg_filtrado = aplicar_filtro(eeg_crudo, fs)

print("Duración (s):", len(eeg_filtrado)/fs)

# SEGMENTACIÓN + PSD después del filtro
segmentos = segmentar(eeg_crudo, fs, duracion=4)
print("Cantidad de segmentos:", len(segmentos))
f, psd_segs = psd_segmentos(segmentos, fs)
psd_media_s_filtro = np.mean(psd_segs, axis=0)

# =========================
# GRAFICO 1 - Estimación espectral de la señal con ruido
# =========================
plt.figure(figsize=(10,6))

for ch in range(4):
    plt.semilogy(f, psd_media_s_filtro[:, ch], color=colores[ch], label=canales[ch])

#plt.title("PSD promedio señal s/ filtro")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.xlim(0, 100)
plt.grid()
plt.legend()
plt.show()

# =========================
# GRAFICO 2 - Estimación espectral de la señal con ruido y detección del 95%
# =========================
resultados = analizar_psd_existente(f, psd_media_s_filtro, canales, plot=True)
#for r in resultados:
#    print(r) 
    
# =========================
# GRAFICO 3 - Señal con ruido y filtrada (trabajo OK)
# =========================

plt.figure(figsize=(12, 8))

for ch in range(4):
    plt.subplot(4,1,ch+1)
    plt.plot(t[:2000], eeg_crudo[:2000, ch], label="Crudo", alpha=0.6)
    plt.plot(t[:2000], eeg_filtrado[:2000, ch], linestyle='--', label="Filtrado")
    plt.title(f"Crudo vs Filtrado - {canales[ch]}")
    plt.grid()
    if ch == 0:
        plt.legend()

plt.xlabel("Tiempo [s]")
plt.tight_layout()
plt.show()

# =========================
# GRAFICO 4 - Señal con ruido y filtrada  en este trabajo
# =========================
plt.figure(figsize=(14, 10))  # más grande para que se vea bien

for ch in range(4):

    # =========================
    # COLUMNA IZQUIERDA: CRUDO
    # =========================
    plt.subplot(4, 2, 2*ch + 1)
    plt.plot(t[2000:3000], eeg_crudo[2000:3000, ch], color=colores[ch], alpha=0.8)
    plt.title(f"Señal original - {canales[ch]}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.grid()

    # =========================
    # COLUMNA DERECHA: FILTRADO
    # =========================
    plt.subplot(4, 2, 2*ch + 2)
    plt.plot(t[2000:3000], eeg_filtrado[2000:3000, ch], color=colores[ch], alpha=0.8)
    plt.title(f"Señal Filtrada - {canales[ch]}")
    plt.grid()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
plt.tight_layout()
plt.show()

# =========================
# GRAFICO 5 - Con la segmentación de ventanas de 4 segundos tenemos multiples PSD de las cuales obtenemos una PSD media para cada electrodo)
# =========================

# SEGMENTACIÓN + PSD después del filtro
segmentos = segmentar(eeg_filtrado, fs, duracion=4)
print("Cantidad de segmentos:", len(segmentos))
f, psd_segs = psd_segmentos(segmentos, fs)
psd_media_c_filtro = np.mean(psd_segs, axis=0)

resultados = analizar_psd_existente(f, psd_media_c_filtro, canales, plot=False)
for r in resultados:
        print(r)

plt.figure(figsize=(10,6))

for ch in range(4):
    plt.semilogy(f, psd_media_c_filtro [:, ch], color=colores[ch], label=canales[ch])

#plt.title("PSD promedio señal c/ filtro")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.xlim(0, 100)
plt.ylim(1e-8, 1e2)
plt.grid()
plt.legend()
plt.show()
# =========================
# GRAFICO 6: ALGUNOS SEGMENTOS (VER VARIABILIDAD)
# =========================

plt.figure(figsize=(10,6))

for i in range(min(5, len(psd_segs))):
    plt.semilogy(f, psd_segs[i, :, 0], alpha=0.5)

#plt.title("PSD de segmentos individuales (Canal T7)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.xlim(0, 60)
plt.ylim(1e-8, 1e2)
plt.grid()

plt.show()
# =========================
# GRAFICO 7: DIFERENCIACIÓN DE BANDAS
# =========================
res = analizar_bandas_psd(f, psd_media_c_filtro, canales, bandas, colores, colores_bandas, plot=True)
# TABLA DE PORCENTAJES
df = pd.DataFrame(res).T  # .T para que canales sean filas
print(df)
# %% DATAFRAME DE LAS 10 PERSONAS

# =========================
# Loop para procesar los datos de las 10 personas
# =========================

resultados_totales = {}

for persona, archivos in personas.items():
    for archivo in archivos:

        eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)

        f, psd_media, energias = procesar_eeg_psd_media(
            eeg, fs, canales, bandas
        )

        resultados_totales[(persona, archivo)] = energias

# =========================
# Dataframe final 
# =========================       

rows = [] #crea lista de filas vacias 

for (persona, archivo), datos_canales in resultados_totales.items():
    for canal, bandas_dict in datos_canales.items():
        row = {
            "Persona": persona,
            "Archivo": archivo,
            "Canal": canal,
            **bandas_dict
        }
        rows.append(row)

df_long = pd.DataFrame(rows)
# =========================
# Extraer situación desde el nombre del archivo
# =========================

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
# %% Boxplot generado con datos de las 10 personas para 4 situaciones y 4 electrodos

# =========================
# Boxplot 
# =========================
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

    ax.set_ylim(0,100)

    # solo el último gráfico muestra etiquetas X
    if j == len(canales)-1:
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=0)
    else:
        ax.set_xticks([])

    ax.set_title(f"({chr(97+j)})", fontsize=11)

    ax.set_ylabel("Energía [%]")

    ax.grid(axis="y", alpha=0.3)


fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

#fig.suptitle("Distribución de energía relativa por banda EEG", fontsize=14)

plt.tight_layout(rect=[0,0.05,1,0.96])

plt.show()
df_long["Situacion"] = df_long["Archivo"].apply(extraer_situacion)
# %%# medianas
# =========================
# Tomamos referencia
# df_medianas = df_long.groupby(["Canal","Situacion"])[["Delta","Theta","Alpha","Beta"]].median()

# print(df_medianas.round(2))
# for canal in canales:
    
#     print(f"\n=== {canal} ===")
    
#     df_canal = df_medianas.loc[canal]
    
#     print(df_canal.round(2))
# %% Boxplot normalizado respecto a la referencia

df_norm = []

for persona in df_long["Persona"].unique():
    
    df_p = df_long[df_long["Persona"] == persona]
    
    for canal in canales:
        
        df_pc = df_p[df_p["Canal"] == canal]
        
        # referencia de esa persona y canal
        ref = df_pc[df_pc["Situacion"] == "Referencia"]
        
        if ref.empty:
            continue
        
        ref_vals = ref[["Delta","Theta","Alpha","Beta"]].values[0]
        
        # recorrer otras situaciones
        for _, row in df_pc.iterrows():
            
            if row["Situacion"] == "Referencia":
                continue
            
            new_row = row.copy()
            
            new_row["Delta"] = row["Delta"] - ref_vals[0]
            new_row["Theta"] = row["Theta"] - ref_vals[1]
            new_row["Alpha"] = row["Alpha"] - ref_vals[2]
            new_row["Beta"]  = row["Beta"]  - ref_vals[3]
            
            df_norm.append(new_row)

df_norm = pd.DataFrame(df_norm)

# =========================
# NUEVAS SITUACIONES (sin referencia)
# =========================

situaciones = ["Música nativa", "Música extranjera", "Música neutra"]
situaciones_abrev = ["MN", "ME", "NEU"]

# =========================
# BOXPLOT NORMALIZADO
# =========================

fig, axs = plt.subplots(4,1, figsize=(10,12), sharey=True)

for j, canal in enumerate(canales):

    ax = axs[j]

    data = []
    labels = []
    colores = []

    for banda in bandas:

        for i, situacion in enumerate(situaciones):

            datos = df_norm[
                (df_norm["Canal"] == canal) &
                (df_norm["Situacion"] == situacion)
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

    # línea de referencia (0)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_ylim(-50, 50)  # ajustable según datos

    if j == len(canales)-1:
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks([])

    ax.set_title(f"({chr(97+j)})", fontsize=11)
    ax.set_ylabel("Δ Energía [%]")
    ax.grid(axis="y", alpha=0.3)

# leyenda de bandas
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False
)

plt.tight_layout(rect=[0,0.05,1,0.96])
plt.show()

# %% Espectrogramas
# --- PARÁMETROS ---
personas_comparar = ["P2", "P4", "P6", "P8", "P10"]
window = 'hann'
nperseg = int(2*fs)
noverlap = int(nperseg/2)
f_max=40
# %% # ESPECTROGRAMA SITUACIÓN: Referencia

situacion = "Referencia"

# --- CARGAR LOS DATOS ---
data = []

for persona in personas_comparar:
    archivo_obj = None
    for archivo in personas[persona]:
        if extraer_situacion(archivo) == situacion:
            archivo_obj = archivo
            break
    if archivo_obj is None:
        continue
    eeg, _ = cargar_eeg(f"{base_path}\\{archivo_obj}", fs)
    eeg = aplicar_filtro(eeg, fs)
    eeg = eeg[:, [0,1,2,3]]  # T7, F8, Cz, P4
    data.append(eeg.T)  # (4, N)

data = np.array(data)  # shape: (5,4,N)

# --- PLOTEO ---
fig, axs = plt.subplots(5, 4, figsize=(16,12), constrained_layout=True)

for p in range(len(personas_comparar)):
    for ch in range(len(canales)):
        x = data[p, ch, :]
        f, t, Sxx = signal.spectrogram(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        Sxx = np.where(Sxx==0, np.finfo(float).eps, Sxx)

        im = axs[p, ch].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        axs[p, ch].set_xlim(50, 65)
        axs[p, ch].set_ylim(0, f_max)
        if p == 0:
            axs[p, ch].set_title(canales[ch])
        if ch == 0:
            axs[p, ch].set_ylabel(f'P{p+1}\nHz')

# --- COLORBAR COMÚN ---
im = axs[p, ch].pcolormesh(
    t, f, 10*np.log10(Sxx),
    shading='gouraud',
    cmap='turbo',   # o 'jet'
    vmin=-20,
    vmax=25
)
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('PSD [dB]')

plt.suptitle(f'Espectrogramas - Situación: {situacion}', fontsize=16)
plt.show()
# %% # ESPECTROGRAMA SITUACIÓN: MÚSICA NATIVA
situacion = "Música nativa"

# --- CARGAR LOS DATOS ---
data = []

for persona in personas_comparar:
    archivo_obj = None
    for archivo in personas[persona]:
        if extraer_situacion(archivo) == situacion:
            archivo_obj = archivo
            break
    if archivo_obj is None:
        continue
    eeg, _ = cargar_eeg(f"{base_path}\\{archivo_obj}", fs)
    eeg = aplicar_filtro(eeg, fs)
    eeg = eeg[:, [0,1,2,3]]  # T7, F8, Cz, P4
    data.append(eeg.T)  # (4, N)

data = np.array(data)  # shape: (5,4,N)

# --- PLOTEO ---
fig, axs = plt.subplots(5, 4, figsize=(16,12), constrained_layout=True)

for p in range(len(personas_comparar)):
    for ch in range(len(canales)):
        x = data[p, ch, :]
        f, t, Sxx = signal.spectrogram(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        Sxx = np.where(Sxx==0, np.finfo(float).eps, Sxx)

        im = axs[p, ch].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        axs[p, ch].set_xlim(50, 65)
        axs[p, ch].set_ylim(0, f_max)
        if p == 0:
            axs[p, ch].set_title(canales[ch])
        if ch == 0:
            axs[p, ch].set_ylabel(f'P{p+1}\nHz')

# --- COLORBAR COMÚN ---
im = axs[p, ch].pcolormesh(
    t, f, 10*np.log10(Sxx),
    shading='gouraud',
    cmap='turbo',   # o 'jet'
    vmin=-20,
    vmax=25
)
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('PSD [dB]')
plt.suptitle(f'Espectrogramas - Situación: {situacion}', fontsize=16)
plt.show()
# %% # ESPECTROGRAMA SITUACIÓN: MÚSICA EXTRANJERA
situacion = "Música extranjera"

# --- CARGAR LOS DATOS ---
data = []

for persona in personas_comparar:
    archivo_obj = None
    for archivo in personas[persona]:
        if extraer_situacion(archivo) == situacion:
            archivo_obj = archivo
            break
    if archivo_obj is None:
        continue
    eeg, _ = cargar_eeg(f"{base_path}\\{archivo_obj}", fs)
    eeg = aplicar_filtro(eeg, fs)
    eeg = eeg[:, [0,1,2,3]]  # T7, F8, Cz, P4
    data.append(eeg.T)  # (4, N)

data = np.array(data)  # shape: (5,4,N)

# --- PLOTEO ---
fig, axs = plt.subplots(5, 4, figsize=(16,12), constrained_layout=True)

for p in range(len(personas_comparar)):
    for ch in range(len(canales)):
        x = data[p, ch, :]
        f, t, Sxx = signal.spectrogram(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        Sxx = np.where(Sxx==0, np.finfo(float).eps, Sxx)

        im = axs[p, ch].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        axs[p, ch].set_xlim(50, 65)
        axs[p, ch].set_ylim(0, f_max)
        if p == 0:
            axs[p, ch].set_title(canales[ch])
        if ch == 0:
            axs[p, ch].set_ylabel(f'P{p+1}\nHz')

# --- COLORBAR COMÚN ---
im = axs[p, ch].pcolormesh(
    t, f, 10*np.log10(Sxx),
    shading='gouraud',
    cmap='turbo',   # o 'jet'
    vmin=-20,
    vmax=25
)
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('PSD [dB]')
plt.suptitle(f'Espectrogramas - Situación: {situacion}', fontsize=16)
plt.show()
# %% # ESPECTROGRAMA POR MÚSICA NEUTRA
situacion = "Música neutra"

# --- CARGAR LOS DATOS ---
data = []

for persona in personas_comparar:
    archivo_obj = None
    for archivo in personas[persona]:
        if extraer_situacion(archivo) == situacion:
            archivo_obj = archivo
            break
    if archivo_obj is None:
        continue
    eeg, _ = cargar_eeg(f"{base_path}\\{archivo_obj}", fs)
    eeg = aplicar_filtro(eeg, fs)
    eeg = eeg[:, [0,1,2,3]]  # T7, F8, Cz, P4
    data.append(eeg.T)  # (4, N)

data = np.array(data)  # shape: (5,4,N)

# --- PLOTEO ---
fig, axs = plt.subplots(5, 4, figsize=(16,12), constrained_layout=True)

for p in range(len(personas_comparar)):
    for ch in range(len(canales)):
        x = data[p, ch, :]
        f, t, Sxx = signal.spectrogram(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        Sxx = np.where(Sxx==0, np.finfo(float).eps, Sxx)

        im = axs[p, ch].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        axs[p, ch].set_xlim(50, 65)
        axs[p, ch].set_ylim(0, f_max)
        if p == 0:
            axs[p, ch].set_title(canales[ch])
        if ch == 0:
            axs[p, ch].set_ylabel(f'P{p+1}\nHz')

# --- COLORBAR COMÚN ---
im = axs[p, ch].pcolormesh(
    t, f, 10*np.log10(Sxx),
    shading='gouraud',
    cmap='turbo',   # o 'jet'
    vmin=-20,
    vmax=25
)
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('PSD [dB]')

plt.suptitle(f'Espectrogramas - Situación: {situacion}', fontsize=16)
plt.show()

# %% #MATRICES DE CONECTIVIDAD ENTRE ELECTRODOS

# =========================================================
# MATRICES DE CORRELACIÓN ENTRE ELECTRODOS POR SITUACIÓN
# =========================================================

eeg_filtrados = {}

for persona, archivos in personas.items():
    eeg_filtrados[persona] = {}

    for archivo in archivos:
        eeg, _ = cargar_eeg(f"{base_path}\\{archivo}", fs)
        eeg_f = aplicar_filtro(eeg, fs)

        situacion = extraer_situacion(archivo)

        eeg_filtrados[persona][situacion] = eeg_f
correlaciones_situaciones = {
    "Reposo": [],
    "Música nativa": [],
    "Música extranjera": [],
    "Música neutra": []
}

for persona in eeg_filtrados:

    for situacion, eeg_filtrado in eeg_filtrados[persona].items():

      
        eeg_filtrado = eeg_filtrado[:, [0,1,2,3]]

        matriz_corr = np.corrcoef(eeg_filtrado.T)

        if situacion == "Referencia":
            correlaciones_situaciones["Reposo"].append(matriz_corr)

        elif situacion in correlaciones_situaciones:
            correlaciones_situaciones[situacion].append(matriz_corr)

 #  Para graficar las matrices
for situacion, matrices in correlaciones_situaciones.items():

    if len(matrices) == 0:
        continue

    matriz_prom = np.mean(matrices, axis=0)

    plt.figure(figsize=(6,5))

    im = plt.imshow(matriz_prom, vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlación")

    plt.xticks(range(len(canales)), canales)
    plt.yticks(range(len(canales)), canales)

    plt.title(f"Matriz de correlación - {situacion}")

    #  números dentro de la matriz
    for i in range(len(canales)):
        for j in range(len(canales)):
            plt.text(j, i, f"{matriz_prom[i,j]:.2f}",
                     ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()

# %% #MATRICES DE CONECTIVIDAD FUNCIONAL ESPECÍFICA POR BANDA

calcular_conectividad_banda(eeg_filtrados, fs, "Alpha", canales, bandas)
calcular_conectividad_banda(eeg_filtrados, fs, "Beta", canales, bandas)
calcular_conectividad_banda(eeg_filtrados, fs, "Theta", canales, bandas)
calcular_conectividad_banda(eeg_filtrados, fs, "Delta", canales, bandas)





