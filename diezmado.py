# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 21:15:56 2025

@author: Vale
"""

# -- coding: utf-8 --
"""
Created on Wed Jun  4 20:01:34 2025

@author: Vale
"""


import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

from scipy.signal import find_peaks
import sounddevice as sd
#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)
#%%
# Vamos a diezmar
N= len(wav_data)
M=2
X=wav_data[::M]
# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
import sounddevice as sd
sd.play(wav_data, fs_audio)
# Esto es para probar como funcionar filtrar antes de diezmar igual esta funcion esta optoimizada
# %%Diezmar usando resample_poly (filtra y luego cambia la tasa)
X_filt = sig.resample_poly(wav_data, up=1, down=M) #up=1 xq no estoy interpolando! no agrego 0, down=M xq es el factor al que disminuyo 
fs_diezmado = fs_audio // M
sd.play(X_filt, fs_diezmado)
#%%
import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
# from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

aprox_name = 'butter'
#aprox_name2 = 'cheby1'
# aprox_name = 'cheby2'
#aprox_name = 'ellip'

fs= fs_audio // M  #Hz
nyquist=(fs/2)/(fs/2)
fpass = 1/M 
ripple = 0.5 # dB
fstop = 1/M
atenuacion = 40 # dB

sos=sig.iirdesign(fpass,fstop, ripple, atenuacion, ftype=aprox_name, output='sos', fs=fs)

npoints = fs
w_rad= np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250)) 
w_rad= np.append(w_rad, np.linspace(40, nyquist, 500, endpoint=True))/ (nyquist* np.pi)
# esto mejoro la resolucion en la subir de 0 a 2 
w, hh = sig.sosfreqz(sos, worN=npoints) 
 
# %% Respuesta de Modulo

moduloH=np.abs(hh)
plt.plot(w/np.pi*fs/2,moduloH)
plt.title('Respuesta de Módulo')  
plt.xlabel('Frec')  
plt.ylabel('Módulo de H') 
plt.legend()  
plt.show()


# %% Respuesta de Fase

faseH=np.angle(hh)
plt.plot(w/np.pi*fs/2,faseH)
plt.title('Respuesta de Fase')  
plt.xlabel('Frec')  
plt.ylabel('Fase de H') 
plt.legend()  
plt.show()

#%%

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

# plot_plantilla(filter_type = 'bandpass', fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion)  
plt.legend()  
plt.show()
# %% FILTRADO

# Cargar datos
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
# Plot señal original
plt.figure()
plt.plot(ecg_one_lead)
plt.title('ECG Original')
plt.show()

# Filtrado con el filtro definido (suponiendo sos ya definido)
ecg_filtrado = sig.sosfilt(sos, ecg_one_lead, axis=0)
demora=68 #lo estimamos a ojo contando la cantidad de muestras entre dos picos entes???
#esto corrige la demora todavia te queda la distorsion de fase: no se parece la formas de las dos senales la filtrada y la no filtrada
fig_dpi=150

# Plot señal filtrada
plt.figure()
plt.plot(ecg_filtrado)
plt.title('ECG Filtrado')
plt.show()

#%% Regiones de interes

plt.figure(figsize=(10,5))
plt.plot(ecg_one_lead, label='ECG Original', color='olive')
plt.plot(ecg_filtrado, label='ECG Filtrado',color='pink')
plt.title('ECG Original vs Filtrado')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()


