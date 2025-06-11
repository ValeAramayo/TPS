import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Respuestas al impulso de cada sistema
h_a = np.array([1, 1, 1, 1])         # x(n-3)+x(n-2)+x(n-1)+x(n)
h_b = np.array([1, 1, 1, 1, 1])      # x(n-4)+...+x(n)
h_c = np.array([1, -1])              # x(n)-x(n-1)
h_d = np.array([1, 0, -1])           # x(n)-x(n-2)

# Lista de sistemas
systems = [
    (h_a, 'a) x(n-3)+x(n-2)+x(n-1)+x(n)'),
    (h_b, 'b) x(n-4)+...+x(n)'),
    (h_c, 'c) x(n)-x(n-1)'),
    (h_d, 'd) x(n)-x(n-2)')
]

# Gráficos
for h, title in systems:
    w, H = freqz(h, worN=8000)
    plt.figure(figsize=(10, 4))

    # Módulo
    plt.subplot(1, 2, 1)
    plt.plot(w/np.pi, np.abs(H))
    plt.title(f'Módulo - {title}')
    plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
    plt.ylabel('|H(w)|')
    plt.grid(True)

    # Fase
    plt.subplot(1, 2, 2)
    plt.plot(w/np.pi, np.angle(H))
    plt.title(f'Fase - {title}')
    plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
    plt.ylabel('Fase (rad)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
