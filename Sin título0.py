import sympy as sp
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import hamming, kaiser, blackmanharris
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla, group_delay

# ==============================
# CARGAR ECG
# ==============================

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
demora=200
# Frecuencia de muestreo
fs = 1000
nyquist = fs / 2

# Tamaño de los filtros
cant_coef_lp = 2001
cant_coef_hp = 5301
#%%
 # VENTANAS

# ==============================
# DISEÑO DEL FILTRO PASA BAJOS
# ==============================

filter_type = 'lowpass'
fpass_lp = 35  # Hz
fstop_lp = 48  # Hz con esto llega a atenuar bien a 50 hz
ripple = 0.1   # dB
attenuation = 40  # dB

frecs_lp = [0.0, fpass_lp, fstop_lp, nyquist]
gains_db_lp = [0, -ripple, -attenuation, -100]
gains_lp = 10**(np.array(gains_db_lp)/20)

num_lp = sig.firwin2(cant_coef_lp, frecs_lp, gains_lp, window=('kaiser', 14), fs=fs)

# Visualización del pasa bajos
w_rad = 2 * np.pi * np.linspace(0.1, nyquist, 1000) / fs
w, hh_lp = sig.freqz(num_lp, worN=w_rad)

# plt.figure(1)
# plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh_lp) + 1e-15), label='Pasa bajos')
# plot_plantilla(filter_type='lowpass', fpass=fpass_lp, ripple=1, fstop=fstop_lp, attenuation=attenuation)
# plt.title('Filtro pasa bajos')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud [dB]')
# plt.grid(which='both', axis='both')
# plt.legend()
# plt.show()

# ==============================
# DISEÑO DEL FILTRO PASA ALTOS
# ==============================

filter_type = 'highpass'
fstop_hp = 0.1  # Hz
fpass_hp = 1.0  # Hz

frecs_hp = [0.0, fstop_hp, fpass_hp, nyquist]
gains_db_hp = [-np.inf, -attenuation, -ripple, 0]
gains_hp = 10**(np.array(gains_db_hp)/20)

num_hp = sig.firwin2(cant_coef_hp, frecs_hp, gains_hp, window=('kaiser', 14), fs=fs)

# Visualización del pasa altos
w, hh_hp = sig.freqz(num_hp, worN=w_rad)

# plt.figure(2)
# plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(hh_hp) + 1e-15), label='Pasa altos')
# plot_plantilla(filter_type='highpass', fpass=fpass_hp, ripple=1, fstop=fstop_hp, attenuation=attenuation, fs=fs)
# plt.title('Filtro pasa altos')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud [dB]')
# plt.grid(which='both', axis='both')
# plt.legend()
# plt.show()

# ==============================
# CONVOLUCIÓN: FILTRO PASA BANDA
# ==============================

num_total = np.convolve(num_lp, num_hp)
w, h_total = sig.freqz(num_total, worN=w_rad)

# plt.figure(3)
# plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(h_total) + 1e-15), label='Filtro combinado (pasa banda)')
# plot_plantilla(filter_type='bandpass', fpass=(fpass_hp, fpass_lp), ripple=1, fstop=(fstop_hp, fstop_lp), attenuation=attenuation, fs=fs)
# plt.title('Filtro resultante (Pasa Banda)')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud [dB]')
# plt.grid(which='both', axis='both')
# plt.legend()
# plt.show()

# ==============================
# FILTRADO
# ==============================

ecg_filtrado = sig.lfilter(num_total, 1, ecg_one_lead) #1 porque es un filt
plt.plot(ecg_filtrado)
plt.plot(ecg_one_lead)
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
# %%
   # CUADRADOS MÍNIMOS

# %%
   # PM - REMEZ
