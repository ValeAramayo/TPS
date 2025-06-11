# -- coding: utf-8 --
"""
Created on Thu May 22 18:25:12 2025

@author: Valentina
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

aprox_name = 'butter'
#aprox_name2 = 'cheby1'
# aprox_name = 'cheby2'
#aprox_name = 'ellip'

fs= 1000 #Hz
nyquist=fs/2
fpass = np.array([1.0, 35.0])
ripple = 0.5 # dB
fstop = np.array([.1, 50.])
atenuacion = 40 # dB

sos=sig.iirdesign(fpass,fstop, ripple, atenuacion, ftype=aprox_name, output='sos', fs=fs)

npoints = 1000 
w_rad= np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250)) 
w_rad= np.append(w_rad, np.linspace(40, nyquist, 500, endpoint=True))/ (nyquist* np.pi)
# esto mejoro la resolucion en la subir de 0 a 2 
w, hh = sig.sosfreqz(sos, worN=npoints)  
plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15), label='sos')  
#1e-15 para no tener problemas de cero
# w/np.pi*fs/2 de cero a nyquist
  

plt.title('Plantilla de diseño')  
plt.xlabel('Frecuencia normalizada a Nyq [#]')  
plt.ylabel('Amplitud [dB]')  
plt.grid(which='both', axis='both')  

ax = plt.gca()  
# ax.set_xlim([0, 1])  
# ax.set_ylim([-60, 10])  

plot_plantilla(filter_type = 'bandpass', fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion, fs=fs)  
plt.legend()  
plt.show()



# FILTRADO
# Cargar datos
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()

# Plot señal original
plt.figure()
plt.plot(ecg_one_lead)
plt.title('ECG Original')
plt.show()

# Filtrado con el filtro definido (suponiendo sos ya definido)
ecg_filtrado = sig.sosfilt(sos, ecg_one_lead)

# Plot señal filtrada
plt.figure()
plt.plot(ecg_filtrado)
plt.title('ECG Filtrado')
plt.show()



plt.figure(figsize=(10,5))
plt.plot(ecg_one_lead, label='ECG Original', color='olive')
plt.plot(ecg_filtrado, label='ECG Filtrado',color='pink')
plt.title('ECG Original vs Filtrado')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()

#%%


fs= 400 #Hz
nyquist=fs/2
fpass = np.array([1, 10])
ripple = 0.5 # dB
fstop = np.array([.1,20])
atenuacion = 80 # dB

sos2=sig.iirdesign(fpass,fstop, ripple, atenuacion, ftype=aprox_name, output='sos', fs=fs)

npoints = 400
w_rad= np.append(np.logspace(-2, 0.8, 500), np.logspace(0.9, 1.6, 500)) 
w_rad= np.append(w_rad, np.linspace(20, nyquist, 200, endpoint=True))/ (nyquist* np.pi)
# esto mejoro la resolucion en la subir de 0 a 2 
w, hh = sig.sosfreqz(sos2, worN=npoints)  
plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15), label='sos2')  
#1e-15 para no tener problemas de cero
# w/np.pi*fs/2 de cero a nyquist
  

plt.title('Plantilla de diseño')  
plt.xlabel('Frecuencia normalizada a Nyq [#]')  
plt.ylabel('Amplitud [dB]')  
plt.grid(which='both', axis='both')  
# Cargar datos ppg
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)

# Plot señal original
plt.figure()
plt.plot(ppg)
plt.title('PPG Original')
plt.show()

# Filtrado con el filtro definido (suponiendo sos ya definido)
ppg_filtrado = sig.sosfilt(sos2, ppg)

# Plot señal filtrada
plt.figure()
plt.plot(ppg_filtrado)
plt.title('PPG Filtrado')
plt.show()



plt.figure(figsize=(10,5))
plt.plot(ppg, label='PPG Original', color='olive')
plt.plot(ppg_filtrado, label='PPG Filtrado',color='pink')
plt.title('PPG Original vs Filtrado')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()
