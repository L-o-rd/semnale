import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def cosinus(A, f0, phi):
    def x(t):
        return A * np.cos(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    tstart = 0.0
    tstop = 1.0

    time = np.linspace(start = tstart, stop = tstop, num = 10 ** 3)
    f0 = 5.0 / (tstop - tstart)

    y = cosinus(1, f0, -np.pi / 2)
    x = sinus(1, f0, 0)

    fig, axs = plt.subplots(2)
    fig.suptitle('Exercitiul 1')
    axs[1].set_title('Cosinus (- Pi / 2)', fontstyle='italic')
    axs[0].set_title('Sinus', fontstyle='italic')
    axs[0].plot(time, x(time))
    axs[1].plot(time, y(time))
    plt.savefig('./labs/02/01.pdf')
    plt.show()