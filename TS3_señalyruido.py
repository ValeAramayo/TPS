# -*- coding: utf-8 -*- 
"""
Created on Fri Apr  4 12:50:15 2025

@author: Usuario
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

##normalizar para que la potencia sea 1
##uno es viendo la varianza pero no 
tt, xx = mi_funcion_sen(1.4, 0, 1, 0, 1000, 1000)
##print (np.var(xx)) #Imprime la varianza de la funcion
##con desvio estandar:
xn=xx/np.std(xx)

plt.figure(1)
plt.plot(tt,xx, label="Señal limpia normalizada") 
plt.title("Señal limpia normalizada:")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

#%% Datos de la simulación

fs =  1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
# con 1000 para cada una normalizamos la resolucion espectral

# Datos del ADC
B = 4 # bits 
Vf = 2 # rango simétrico de +/- Vf Volts 
q = 2*Vf/(2**B)# paso de cuantización de q Volts

##1 de ganancia, fijarte el ancho de banda, y la potencia del radio 50 al cuadrado
# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 


ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral
#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform

nn=np.random.normal(0,np.sqrt(pot_ruido_analog),N) #señal de ruido analogico

# Señales

analog_sig = xn # señal analógica sin ruido

sr = xn+nn # señal analógica de entrada al ADC (con ruido analógico)
plt.figure(2)
plt.plot(tt,sr, label="Señal analógica con ruido")
plt.title("Señal analogica con ruido:")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

srq = np.round(sr/q)*q# señal cuantizada, (señal divida la cantidad total de bits)
                                 
nq =  srq-sr# señal de ruido de cuantización

plt.figure(3)
plt.plot(tt,nq, label="Ruido de cuantización")
plt.title("Señal analogica con ruido:")
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud")
plt.legend()
plt.show()

#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=1, linestyle='-', color='blue', fillstyle='none', label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# Línea punteada (sin marcadores)
plt.plot(tt, sr, lw=2, color='green', linestyle='dotted', label='$ s_R = s + n $')

# Círculos sin relleno cada N puntos (por ejemplo, cada 10)
plt.plot(tt[::2], sr[::2], marker='o', markersize=2, linestyle='none', markerfacecolor='none', 
         markeredgecolor='green', label='Muestras de $s_r$')
plt.plot(tt, xx, color='yellow', linestyle=':', label='$s$ (analog)')

plt.legend()
plt.grid(True)
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Señales: original, ruidosa y salida del ADC')
plt.show()


###########
# Espectro
###########

plt.figure(2)
plt.ylim(-400, 0)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten(), bins=bins, alpha=0.5)
e=q*0.5
plt.plot([-e, -e, e, e], [0, N/bins, N/bins, 0], '--r')
plt.title(f'Ruido de cuantización (B={B}, q={q:.3f} V)')
plt.ylabel('Frecuencia')
plt.xlabel('Error de cuantización (V)')
plt.show()
