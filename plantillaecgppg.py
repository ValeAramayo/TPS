# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:52:28 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt

# Eje de frecuencia normalizada (de 0 a π)
w = np.linspace(0, np.pi, 1024)
z = np.exp(1j * w)

# Definición de filtros (transformadas Z evaluadas en e^{jω})
Hz = {
    'Filtro a': z*-3 + z-2 + z*-1 + 1,
    'Filtro b': z*-4 + z-3 + z-2 + z*-1 + 1,
    'Filtro c': 1 - z**-1,
    'Filtro d': 1 - z**-2
}

# --- Respuesta de módulo (sin dB) ---
plt.figure(figsize=(10, 4))
for label, H in Hz.items(): #recorre todos los filtros guardados en Tz, evluando en Z=e^{jω} asi obtengo la respuesta en frecuencia (T)
    plt.plot(w / np.pi, np.abs(H), label=label)
plt.title('Respuesta en Magnitud |H(e^{jω})|')
plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
plt.ylabel('Magnitud (valor absoluto)')
plt.grid()
plt.legend()
plt.tight_layout()
#%% Respuesta fase
plt.figure(figsize=(10, 4))
for label, H in Hz.items():
    plt.plot(w/np.pi, np.angle(H), label=label)
plt.title('Respuesta en Fase ∠H(e^{jω})')
plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()