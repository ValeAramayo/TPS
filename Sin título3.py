# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:07:25 2025

@author: Vale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from pytc2.sistemas_lineales import plot_plantilla, group_delay
# Parámetros
fs = 1000             # Frecuencia de muestreo en Hz
nyquist = fs / 2
cant_coef_hp = 100001 # Cantidad de coeficientes (impar -> simétrico para fase lineal)

# Especificaciones de plantilla
fstop = 0.1          # Frecuencia de stopband (Hz)
fpass = 1.0          # Frecuencia de passband (Hz)
attenuation = 40     # Atenuación mínima en banda de detención (dB)
ripple = 0.1          # Ripple máximo en banda pasante (dB)

# Frecuencias y ganancias
frecs = [0.0, fstop, fpass, nyquist]
gains_db = [-np.inf, -attenuation, -ripple, 0]
gains = 10 ** (np.array(gains_db) / 20)  # Conversión de dB a magnitud lineal

# Diseño FIR con Kaiser
num_ka = sig.firwin2(cant_coef_hp, frecs, gains, window=('kaiser', 14), fs=fs)

# Frecuencias para graficar (en Hz y radianes/muestra)
w_hz = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250))
w_hz = np.append(w_hz, np.linspace(40, nyquist, 500))
w_rad = 2 * np.pi * w_hz / fs

# Respuesta en frecuencia
_, hh = sig.freqz(num_ka, worN=w_rad)

# Gráfico de la magnitud
plt.figure(figsize=(10, 6))
plt.plot(w_hz, 20 * np.log10(np.maximum(np.abs(hh), 1e-10)), label='FIR Kaiser')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Filtro FIR Pasa-Altos (Kaiser)')
plt.grid(True, which='both')
plt.xscale('log')
plt.ylim([-100, 5])

# Plantilla visual
plt.axvline(fstop, color='red', linestyle='--', label='fstop = 0.1 Hz')
plt.axvline(fpass, color='green', linestyle='--', label='fpass = 1 Hz')
plt.axhline(-attenuation, color='red', linestyle='--', label='Atenuación = -40 dB')
plt.axhline(-ripple, color='green', linestyle='--', label='Ripple = -1 dB')
plt.legend()
plt.tight_layout()
plt.show()
plot_plantilla(filter_type='highpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
