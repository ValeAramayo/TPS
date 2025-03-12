#%% DESCRIPCIÓN
# Generador de señales: Senoidales
#  En este programa vamos a parametrizar y llamar a una función "funcion_senoidal"

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np # La uso para realizar operaciones matemáticas como el seno y para hacer la linea de tiempo tt
import matplotlib.pyplot as plt # La uso para imprimir el gráfico del seno

#%% Defino la función funcion_senoidal
def funcion_senoidal(vmax,dc,ff,ph,nn,fs): 
  # funcion_senoidal(vMax,vDc,frec,fase,#_muestras,frec_muestreo)
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    # otra forma: tt = np.arange(start = 0, stop = T_simulacion, step = Ts)
   
    xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    
    return tt, xx
    
#%% Llamo a la función función_senoidal

fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
tt, xx = funcion_senoidal (vmax = 2, dc = 0, ff = 1, ph=0, nn = N, fs = fs)

plt.figure(1)


plt.plot(tt,xx,)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.title('Señal Senoidal')
plt.grid()
plt.show()
