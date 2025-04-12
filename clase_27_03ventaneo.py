#%% Módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift  # Corrección en la importación

def mi_funcion_sen(vmax, dc, ff, ph, N, fs):
    ts = 1/fs  # Tiempo de muestreo o período
    tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempo
    
    # Generación de la señal senoidal
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    
    return tt, xx

#%% Datos de la simulación

fs = 1000  # Frecuencia de muestreo (Hz)
N = 1000   # Cantidad de muestras

# Datos del ADC
B = 8  # Bits de resolución
Vf = 2  # Rango simétrico de +/- Vf Volts
q = 2 * Vf / (2**B)  # Paso de cuantización de q Volts

# Ruido de cuantización y analógico
pot_ruido_cuant = q**2 / 12  
kn = 1.  
pot_ruido_analog = pot_ruido_cuant * kn  

df = fs/N  # Resolución espectral

# Generación de la señal
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=N/4, ph=0, N=N, fs=fs)

# Aplicar ventana coseno
window = sig.windows.cosine(N)
xw = xx * window  # Señal ventaneada
xn = xx / np.std(xx)  # Normalización

# Definir correctamente Xw_1
Xw_1 = xw / np.std(xw)

# Graficar señal limpia
plt.figure()
plt.plot(tt, xx, label="Señal sin ventana")
plt.plot(tt, xw, label="Señal con ventana", linestyle='dashed')
plt.title("Señal con y sin ventana")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

# Definir señales
analog_sig = Xw_1
analog_sig_1 = xn

#%% Ruido y cuantización
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido analógico

sr = xn + nn  # Señal con ruido
srq = np.round(sr/q) * q  # Cuantización

# Ruido de cuantización
nq = srq - sr

# Graficar ruido de cuantización
plt.figure()
plt.plot(tt, nq)
plt.title("Ruido de cuantización")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.show()

#%% Espectro con y sin ventana
plt.figure()
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2

ft_As = 1/N * np.fft.fft(analog_sig)
ft_As_1 = 1/N * np.fft.fft(analog_sig_1)

plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_As[bfrec])**2), label='Con ventana', color='red', linestyle='dotted')
plt.plot(ff[bfrec], 10*np.log10(2*np.abs(ft_As_1[bfrec])**2), label='Sin ventana', color='blue', linestyle='dotted')

plt.title("Espectro con y sin ventana")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend()
plt.show()
