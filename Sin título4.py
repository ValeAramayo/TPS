
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:52:30 2025

@author: Vale
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla

# frecuencia de muestreo
fs = 1000
nyquist = fs / 2

# diseño del filtro pasa banda
cant_coef_firls = 1501
cant_coef_remez = 2501

fpass = np.array([1., 35.])   
fstop = np.array([0.1, 50.])  
ripple = 0.5  # dB
attenuation = 40  # dB

# Bandas corregidas (sin valores repetidos)
f = np.array([0, fstop[0], fpass[0], fpass[1], fstop[1], nyquist])
hh = np.array([0, 0, 1, 1, 0, 0])

# Diseño con firls
Filtro_firls = sig.firls(numtaps=cant_coef_firls, bands=f, desired=hh, fs=fs)
w, hh_firls = sig.freqz(Filtro_firls, worN=1000)

plt.figure()
plt.plot(w / np.pi * nyquist, 20 * np.log10(np.abs(hh_firls) + 1e-15), label='FIRLS')
plt.title('Filtro FIRLS')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.legend()
plt.show()

# Diseño con remez
Filtro_remez = sig.remez(numtaps=cant_coef_remez, bands=f, desired=hh, fs=fs)
w, hh_remez = sig.freqz(Filtro_remez, worN=1000)

plt.figure()
plt.plot(w / np.pi * nyquist, 20 * np.log10(np.abs(hh_remez) + 1e-15), label='Remez')
plt.title('Filtro Remez')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.legend()
plt.show()
