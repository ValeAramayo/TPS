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




N2=N*1000
#FETEAR LA MUESTRA n/4

final_fft_1=1/N*np.fft.fft(xk_con_ventana, n=N2, axis=0) # n=N2
final_fft_2=1/N*np.fft.fft(xk_con_ventana2,n=N2, axis=0)
final_fft_3=1/N*np.fft.fft(xk_sin_ventana, n=N2, axis=0)

final_BMH = np.abs(final_fft_1)
final_FLT= np.abs(final_fft_2)
final_BOX = np.abs(final_fft_3)

# indice= N/4
k_1= np.argmax(final_BMH[:N//250, ;], axis=0) # vector para quitar la feta
k_2= np.argmax(final_BMH[:N//250, ;], axis=0)
k_3= np.argmax(final_BMH[:N//250, ;], axis=0)
# A_GORRO = np.array([a_gorro_1, a_gorro_2, a_gorro_3])  # También da (3, 200)

estimadoromega1= k_1*df
estimadoromega2= k_2*df
estimadoromega3= k_3*df
# # Etiquetas para cada conjunto
# labels = ['Blackman-Harris', 'Flattop', 'Boxcar']


# # Graficar los 3 histogramas superpuestos
# plt.figure(1)
# plt.hist(a_gorro_1, bins=30, label='Blackman-Harris', color='blue', alpha=0.6)
# plt.hist(a_gorro_2, bins=30, label='Flattop', color='green', alpha=0.6)
# plt.hist(a_gorro_3, bins=30, label='Boxcar', color='red', alpha=0.6)


# plt.title('Histogramas de Magnitud FFT (una por ventana)')
# plt.xlabel('Magnitud')
# plt.ylabel('Frecuencia')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# E1= np.mean(a_gorro_1) #esperanza
# E2= np.mean(a_gorro_2)
# E3= np.mean(a_gorro_3)

# #
# sesgo1= E1 - a1 
# sesgo2= E2 - a1
# sesgo3=E3 - a1

# V1= np.var(a_gorro_1) #varianza
# V2= np.var(a_gorro_2)
# V3= np.var(a_gorro_3)



#
Estimador_omega1= np.argmax(final_BMH[:N2//2, :],axis=0)

