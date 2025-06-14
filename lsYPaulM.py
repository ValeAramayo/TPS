# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:52:30 2025

@author: Vale
"""
import sympy as sp
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import hamming, kaiser, blackmanharris
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla, group_delay

# frecuencia de muestreo normalizada
fs = 1000
nyquist=fs/2
# tamaño de la respuesta al impulso
cant_coef = 1501 #explicar pq van coeficientes impares
filter_type = 'bandpass'

fpass = np.array([1., 35.])   
fstop = np.array([.1, 50.])  
ripple = 0.5 # dB
attenuation = 40 # dB

# construyo la plantilla de requerimientos
f = np.array([0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyquist]) # fpass[1]+1
hh = np.array([0, 0, 1, 1, 0, 0])

# Diseño del filtro
Filtro_cuadrado= sig.firls(numtaps = 1501, bands = f, desired = hh, fs = 1000)
# Es necesario tener orden impar!! Estudiar esto de los filtros fir
npoints=1000
# hh = sig.firls(numtaps = 1501, bands = f, desired = hh, fs = 1000)
w, hh_firls = sig.freqz(Filtro_cuadrado, worN=npoints) # <interpoló> los puntos obtenidos

plt.figure()
plt.plot(w/np.pi * fs / 2, 20*np.log10(np.abs(hh_firls )+1e-15), label='Firls')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type='bandpass', fpass=(fpass[0], fpass[1]), ripple=1, fstop=(fstop[0], fstop[1]), attenuation=attenuation, fs=fs)
plt.legend()
plt.show()

 
# # Diseño del filtro
Filtro_Remez = sig.remez(numtaps = 2501, bands = f, desired=hh[::2], fs = 1000)
w, hh_remez = sig.freqz(Filtro_Remez, worN=npoints)
plt.figure()
plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh_remez) + 1e-15), label='remez')
plt.title('Plantilla de diseño con Remez')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type='bandpass', fpass=(fpass[0], fpass[1]), ripple=1, fstop=(fstop[0], fstop[1]), attenuation=attenuation, fs=fs)
plt.legend()
plt.show()
