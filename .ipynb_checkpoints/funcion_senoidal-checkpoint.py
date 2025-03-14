#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 12 2025

@author: Valentina Aramayo

Descripción:
------------

    Generador de señales: Senoidales
    En este programa vamos a parametrizar y llamar a una función "funcion_senoidal"
    Luego, se realizarán pruebas com distintas frecuencias f0 (en este código ff)
    Finalmente, se implementará una señal triangular
"""
#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np # La uso para realizar operaciones matemáticas como el seno y para hacer la linea de tiempo tt
import matplotlib.pyplot as plt # La uso para imprimir el gráfico del seno
from scipy.signal import sawtooth # Importar señal triangular

#%% Defino la función funcion_senoidal
def funcion_senoidal(vmax,dc,ff,ph,nn,fs): 
  # funcion_senoidal(vMax,vDc,frec,fase,#_muestras,frec_muestreo)
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    # otra forma: tt = np.arange(start = 0, stop = T_simulacion, step = Ts)
   
    xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    plt.figure()
    plt.plot(tt,xx,)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud  [V]')
    plt.title('Señal Senoidal')
    plt.grid()
    plt.show()
    return tt, xx  

#%% Defino la función funcion_triangular
def funcion_triangular(vmax,dc,ff,ph,nn,fs): 
  # funcion_senoidal(vMax,vDc,frec,fase,#_muestras,frec_muestreo)
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    # otra forma: tt = np.arange(start = 0, stop = T_simulacion, step = Ts)
   
    xx = vmax * sawtooth(2 * np.pi * ff * tt + ph, width=0.5) + dc
    plt.figure()
    plt.plot(tt,xx,)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.title('Señal Triangular')
    plt.grid()
    plt.show()
    return tt, xx  
#%% Parámetros fs y N
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
#%% Experimento 1

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 1, ph=0, nn = N, fs = fs)

#%% Experimento 2

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 500, ph=0, nn = N, fs = fs)
#%% Experimento 3

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 999, ph=0, nn = N, fs = fs)
#%% Experimento 4

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 1001, ph=0, nn = N, fs = fs)
#%% Experimento 5

tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 2001, ph=0, nn = N, fs = fs)
#%% Otra señal
tt, xx = funcion_triangular (vmax = 1, dc = 0, ff =1, ph=0, nn = N, fs = fs)

