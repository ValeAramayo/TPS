# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 02:49:55 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt

def comparar_histogramas_ampliado(B_values, kn_values):
    fs = 1000
    N = 1000
    Vf = 2
    ts = 1/fs
    tt = np.linspace(0, (N-1)*ts, N)
    xx = 1.4 * np.sin(2 * np.pi * 1 * tt)
    xn = xx / np.std(xx)

    fig, axes = plt.subplots(len(kn_values), len(B_values), figsize=(15, 10), sharey=True)
    fig.suptitle('Histograma del ruido de cuantización\nComparación entre niveles de bits y ruido analógico', fontsize=16)

    for i, kn in enumerate(kn_values):
        for j, B in enumerate(B_values):
            q = 2*Vf / (2**B)
            pot_ruido_cuant = q**2 / 12
            pot_ruido_analog = pot_ruido_cuant * kn
            nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)
            sr = xn + nn
            srq = np.round(sr / q) * q
            nq = srq - sr

            ax = axes[i, j]
            ax.hist(nq / q, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
            ax.set_title(f"B={B}, kn={kn}")
            ax.set_xlabel('Error / q')
            if j == 0:
                ax.set_ylabel('Frecuencia')
            ax.grid(True)
            ax.plot([-0.5, -0.5, 0.5, 0.5], [0, N/20, N/20, 0], '--r')  # Uniforme ideal

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Ejecutar con más kn y más B
B_values = [2, 4, 8, 16]
kn_values = [0, 0.01, 0.1, 1, 10]
comparar_histogramas_ampliado(B_values, kn_values)
