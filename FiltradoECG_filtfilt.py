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
# moduloH=np.abs(hh, w/np.pi)
# faseH=np.angle(hh)
# plt.plot(w/np.pi*fs/2,faseH)
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

plot_plantilla(filter_type = 'bandpass', fpass = fpass, ripple = ripple , fstop = fstop, attenuation = atenuacion)  
plt.legend()  
plt.show()





# FILTRADO
# Cargar datos
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
# Plot señal original
# plt.figure()
# plt.plot(ecg_one_lead)
# plt.title('ECG Original')
# plt.show()

# Filtrado con el filtro definido (suponiendo sos ya definido)
ecg_filtrado = sig.sosfiltfilt(sos, ecg_one_lead, axis=0)
demora=0
fig_dpi=150
#%%
regs_interes = ( 
    [4000, 5500], # muestras sin ruido
      [10e3, 11e3], # muestras sin ruido
        np.array([5, 5.2]) *60*fs, # minutos a muestras sin? ruido
        np.array([12, 12.4]) *60*fs, # minutos a muestras con ruido
        np.array([15, 15.2]) *60*fs, # minutos a muestras con ruido
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2,color='olive')
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ecg_filtrado[zoom_region + demora], label='Win',color='pink')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
    

#fijate si un latido se ve que hay pequenas diferencias entre las senales no se puede deber a fase porque ya hicimos filtfilt filtrado bidireccional 
# trade off podrias agrandar un poco mas la banda de paso para que la atenuacion no sea tan abrupta

