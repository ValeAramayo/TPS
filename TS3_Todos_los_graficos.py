def simular_efecto_cuantizacion(kn, B):
    import numpy as np
    import matplotlib.pyplot as plt

    # Datos de la señal
    fs = 1000  # frecuencia de muestreo (Hz)
    N = 1000   # cantidad de muestras
    Vf = 2     # rango simétrico de +/- Vf Volts 
    q = 2 * Vf / (2 ** B)  # paso de cuantización

    # Potencias del ruido
    pot_ruido_cuant = q**2 / 12
    pot_ruido_analog = pot_ruido_cuant * kn

    # Tiempo
    ts = 1 / fs
    tt = np.linspace(0, (N - 1) * ts, N)

    # Señal senoidal y normalización
    xx = 0 + 1.4 * np.sin(2 * np.pi * 1 * tt + 0)
    xn = xx / np.std(xx)

    # Señal con ruido analógico
    nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)
    sr = xn + nn

    # Cuantización
    srq = np.round(sr / q) * q
    nq = srq - sr
    analog_sig = xn

    # Espectros
    df = fs / N
    ff = np.linspace(0, (N - 1) * df, N)
    bfrec = ff <= fs / 2

    ft_SR = 1 / N * np.fft.fft(sr)
    ft_Srq = 1 / N * np.fft.fft(srq)
    ft_As = 1 / N * np.fft.fft(analog_sig)
    ft_Nq = 1 / N * np.fft.fft(nq)
    ft_Nn = 1 / N * np.fft.fft(nn)

    Nnq_mean = np.mean(np.abs(ft_Nq) ** 2)
    nNn_mean = np.mean(np.abs(ft_Nn) ** 2)

    # Plot de señales
    plt.figure()
    plt.plot(tt, srq, lw=1, linestyle='-', color='blue', label='$s_Q$')
    plt.plot(tt, sr, lw=2, color='green', linestyle='dotted', label='$s_R = s + n$')
    plt.plot(tt[::2], sr[::2], marker='o', markersize=2, linestyle='none',
             markerfacecolor='none', markeredgecolor='green', label='Muestras $s_r$')
    plt.plot(tt, xx, color='yellow', linestyle=':', label='$s$ (analog)')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title(f'Señales (B={B}, kn={kn})')
    plt.show()

    # Plot de espectros
    plt.figure()
    if B == 16:
        plt.ylim(-130, 0)
    else:
        plt.ylim(-80, 0)

    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_As[bfrec]) ** 2), color='orange', ls='dotted', label='$s$')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SR[bfrec]) ** 2), ':g', label='$s_R$')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Srq[bfrec]) ** 2), lw=2, label='$s_Q$')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nn[bfrec]) ** 2), ':r')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nq[bfrec]) ** 2), ':c')

    plt.plot([ff[0], ff[bfrec][-1]], [10 * np.log10(2 * nNn_mean)] * 2,
             '--r', label=f'$\overline{{n}}$ = {10 * np.log10(2 * nNn_mean):.1f} dB')
    plt.plot([ff[0], ff[bfrec][-1]], [10 * np.log10(2 * Nnq_mean)] * 2,
             '--c', label=f'$\overline{{n_Q}}$ = {10 * np.log10(2 * Nnq_mean):.1f} dB')

    plt.title(f'Spectro (B={B} bits, q={q:.2e} V)')
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.legend()
    plt.grid()
    plt.show()

    # Histograma del ruido de cuantización
    plt.figure()
    bins = 10
    plt.hist(nq.flatten(), bins=bins, alpha=0.5)
    e = q * 0.5
    plt.plot([-e, -e, e, e], [0, N / bins, N / bins, 0], '--r')
    plt.title(f'Ruido de cuantización (B={B}, q={q:.2e} V)')
    plt.xlabel('Error de cuantización (V)')
    plt.ylabel('Frecuencia')
    plt.grid()
    plt.show()

# Llamadas a la función con distintos parámetros
simular_efecto_cuantizacion(kn=0.1, B=4)
simular_efecto_cuantizacion(kn=1, B=8)
simular_efecto_cuantizacion(kn=1, B=16)
simular_efecto_cuantizacion(kn=10, B=16)
