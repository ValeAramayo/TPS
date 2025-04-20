# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:25:18 2025

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:27:22 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift  # Corrección en la importación


#%% Datos de la simulación
fs = 1000  # Frecuencia de muestreo (Hz)
N = 1000   # Cantidad de muestras

df=fs/N #resolución espectral
ts=1/fs #tiempo de muestro

SNR= 3 #db
R=200 #Número de pruebas (Realizaciones)
omega_0= fs/4 # hay que poner despues fs/4
a1=np.sqrt(2)

#%%
fr= np.random.uniform (-1/2, 1/2, size=(1,R))

omega_1= omega_0 + fr*(df)
tt = np.linspace(0, (N-1)*ts, N).reshape(N, 1)  # (1000, 1)
vtt = np.tile(tt, (1, R))  # (1000, 10)
# Para el zero padding

#%% Señales
#Matriz de senoidales
señal_analogica = a1*np.sin(2*np.pi*omega_1*vtt)  # Dimensión (1000, 10)
#Ruido
potencia_nn=10**(-SNR/10)
sigma= np.sqrt(potencia_nn)
nn= np.random.normal(0, sigma, size=(N,R))

#%% Señal con ruido 
xk= señal_analogica + nn
#%%  Fast Fourier Transform
xk_fft= np.fft.fft(xk, axis=0)/N #hacemos fft por columnas
ff=np.linspace(0, (N-1)*df, N)


#%%  Gráficos
# Definir correctamente la variable
brec = ff <= fs/2  

# Graficar usando el índice correcto
plt.figure(figsize=(10, 5))  # Ajustar tamaño del gráfico

# Graficar todas las realizaciones de una sola vez
plt.plot(ff[brec], 10 * np.log10(2 * np.abs(xk_fft[brec, :])**2)) #xk_fft[brec, :] selecciona todas las columnas para las frecuencias dentro de brec.

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title("Espectros de todas las realizaciones")
plt.grid()
plt.show()

#%% Señales con y sin ventana blackmanharris
ventana = sig.windows.blackmanharris(N).reshape(N, 1)
xk_sin_ventana = señal_analogica + nn
xk_con_ventana = xk_sin_ventana*ventana

#%% FFTs
fft_sin_ventana = np.fft.fft(xk_sin_ventana, axis=0) / N
fft_con_ventana = np.fft.fft(xk_con_ventana, axis=0) / N



#%% Frecuencias
ff = np.linspace(0, (N-1)*df, N)
brec = ff <= fs / 2

#%% Gráficos: ambas curvas en el mismo gráfico

plt.figure(figsize=(12, 6))

# Graficar todas las realizaciones de una sola vez
for i in range(R):
   # plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_sin_ventana[brec, i])**2), alpha=0.4, label='Sin ventana' if i==0 else "")
    plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_con_ventana[brec, i])**2), alpha=0.4, linestyle='--', label='Con ventana' if i==0 else "")

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (dB)")
plt.title("Espectros individuales con y sin ventana (R realizaciones) BMH")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# for i in range(R):
#     plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_sin_ventana[brec, i])**2), color='blue', alpha=0.4, label='Sin ventana' if i==0 else "")
#     plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_con_ventana[brec, i])**2), color='red', alpha=0.4, linestyle='--', label='Con ventana' if i==0 else "")

#%% Señales con y sin ventana flattop
ventana2 = sig.windows.flattop(N).reshape(N, 1)

xk_sin_ventana2 = señal_analogica + nn
xk_con_ventana2 = xk_sin_ventana2*ventana2

#%% FFTs
fft_sin_ventana2 = np.fft.fft(xk_sin_ventana2, axis=0) / N
fft_con_ventana2 = np.fft.fft(xk_con_ventana2, axis=0) / N

#%% Frecuencias
ff = np.linspace(0, (N-1)*df, N)
brec = ff <= fs / 2

#%% Gráficos: ambas curvas en el mismo gráfico

plt.figure(figsize=(12, 6))

# Graficar todas las realizaciones de una sola vez
for i in range(R):
   # plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_sin_ventana2[brec, i])**2), alpha=0.4, label='Sin ventana' if i==0 else "")
    plt.plot(ff[brec], 10 * np.log10(2 * np.abs(fft_con_ventana2[brec, i])**2), alpha=0.4, linestyle='--', label='Con ventana' if i==0 else "")

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (dB)")
plt.title("Espectros individuales con y sin ventana (R realizaciones) flattpop")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#%% Sin zero padding

final_fft_1 = np.fft.fft(xk_sin_ventana, axis=0) / N
final_fft_2 = np.fft.fft(xk_con_ventana, axis=0) / N
final_fft_3 = np.fft.fft(xk_sin_ventana2, axis=0) / N

#Frecuencias
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs / 2
#
final_BMH = np.abs(final_fft_1[bfrec,:])
final_FLT= np.abs(final_fft_2[bfrec,:])
final_BOX = np.abs(final_fft_3[bfrec,:])

# 
omega1_BOX= np.argmax(final_BOX, axis=0)*df # vector para quitar la feta
omega1_BMH= np.argmax(final_BMH, axis=0)*df
omega1_FLT= np.argmax(final_FLT, axis=0)*df

# # Graficar los 3 histogramas superpuestos
plt.figure(1)
plt.hist(omega1_BOX, bins=30, label='Boxcar', color='red', alpha=0.5)
plt.hist(omega1_BMH, bins=30, label='Blackman-Harris', color='blue', alpha=0.5)
plt.hist(omega1_FLT, bins=30, label='Flattop', color='green', alpha=0.5)

plt.title('Histogramas de Frecuencia FFT (una por ventana)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#

sesgo1_a= np.mean(omega1_BOX) -  np.mean(omega_1) 
sesgo2_a= np.mean(omega1_BMH) - np.mean(omega_1)
sesgo3_a= np.mean(omega1_FLT) - np.mean(omega_1 )

V1_a= np.var(omega1_BOX) #Aca deberia ver si se condice con lo que habia dicho el profe que era la varianza
V2_a= np.var(omega1_BMH)
V3_a= np.var(omega1_FLT)

sesgo1_b=np.mean(omega1_BOX - omega_1.flatten())
sesgo2_b= np.mean(omega1_BMH - omega_1.flatten())
sesgo3_b= np.mean(omega1_FLT - omega_1.flatten())
#%%Con zero padding
n=10*N
final_fft_1 = np.fft.fft(xk_sin_ventana, n,axis=0)/ N
final_fft_2 = np.fft.fft(xk_con_ventana, n,axis=0)/ N
final_fft_3 = np.fft.fft(xk_sin_ventana2, n,axis=0)/ N

#Frecuencias
M=10000
df_pad=fs/M
ff_m = np.linspace(0, (M-1)*df_pad, M)
bfrec_m = ff_m <= fs / 2
#
final_BMH_pad = np.abs(final_fft_1[bfrec_m,:])
final_FLT_pad = np.abs(final_fft_2[bfrec_m,:])
final_BOX_pad = np.abs(final_fft_3[bfrec_m,:])


omega1_BOX_pad= np.argmax(final_BOX_pad , axis=0)*df_pad 
omega1_BMH_pad= np.argmax(final_BMH_pad , axis=0)*df_pad
omega1_FLT_pad= np.argmax(final_FLT_pad , axis=0)*df_pad

# # Graficar los 3 histogramas superpuestos
plt.figure(2)
plt.hist(omega1_BOX_pad, bins=30, label='Boxcar', color='red', alpha=0.5)
plt.hist(omega1_BMH_pad, bins=30, label='Blackman-Harris', color='blue', alpha=0.5)
plt.hist(omega1_FLT_pad, bins=30, label='Flattop', color='green', alpha=0.5)

plt.title(f'Histogramas de Frecuencia estimadas (Con zero padding), {R} repeticione, SNR= {SNR}' )
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Sesgo - Forma 1 (Método a)
#Frecuencia de cada realización
F_real = omega_1 / (2 * np.pi)
sesgo1_pad_a= np.mean(omega1_BOX_pad) -  np.mean(omega_1) #Acá no se si tengo que corregir que el vtt tambien sea con el zero padding
sesgo2_pad_a= np.mean(omega1_BMH_pad) - np.mean(omega_1)
sesgo3_pad_a= np.mean(omega1_FLT_pad) - np.mean(omega_1 )

V1_pad_a= np.var(omega1_BOX_pad) #Aca deberia ver si se condice con lo que habia dicho el profe que era la varianza
V2_pad_a= np.var(omega1_BMH_pad)
V3_pad_a= np.var(omega1_FLT_pad)
#Sesgo - Forma 2

sesgo1_pad_b = np.mean(omega1_BOX_pad - omega_1.flatten())
sesgo2_pad_b = np.mean(omega1_BMH_pad - omega_1.flatten())
sesgo3_pad_b = np.mean(omega1_FLT_pad - omega_1.flatten())

V1_pad_b= np.var(omega1_BOX_pad) #Aca deberia ver si se condice con lo que habia dicho el profe que era la varianza
V2_pad_b= np.var(omega1_BMH_pad)
V3_pad_b= np.var(omega1_FLT_pad)

#%% Comparativa de Sesgo y Varianza - Gráfico de barras con leyenda bien mostrada
#Revisar
# ventanas = ['Boxcar', 'Blackman-Harris', 'Flattop']
# sesgos = [sesgo1_a, sesgo2_a, sesgo3_a]
# varianzas = [V1_a, V2_a, V3_a]

# x = np.arange(len(ventanas))  # Posiciones para las barras
# width = 0.35  # Ancho de cada barra

# fig, ax1 = plt.subplots(figsize=(10,6))

# # Barras de sesgo
# b1 = ax1.bar(x - width/2, sesgos, width, color='skyblue', label='Sesgo (Hz)')
# ax1.set_ylabel('Sesgo (Hz)', color='skyblue')
# ax1.tick_params(axis='y', labelcolor='skyblue')

# # Segundo eje Y para varianza
# ax2 = ax1.twinx()
# b2 = ax2.bar(x + width/2, varianzas, width, color='salmon', label='Varianza (Hz²)')
# ax2.set_ylabel('Varianza (Hz²)', color='salmon')
# ax2.tick_params(axis='y', labelcolor='salmon')

# # Títulos y ejes
# ax1.set_xlabel('Ventana')
# ax1.set_title('Comparación de Sesgo y Varianza entre ventanas')
# ax1.set_xticks(x)
# ax1.set_xticklabels(ventanas)
# ax1.grid(True)

# # Leyenda combinada (creamos handles y labels manualmente)
# bars = [b1[0], b2[0]]
# labels = ['Sesgo (Hz)', 'Varianza (Hz²)']
# ax1.legend(bars, labels, loc='upper left')

# plt.tight_layout()
# plt.show()



