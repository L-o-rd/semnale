import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    time = np.linspace(start = 0.0, stop = 1.0, num = 10 ** 5)
    fs = 100

    xs = [
        # sinus(A, f0, phi)

        sinus(1, fs / 2, 0),
        sinus(1, fs / 4, 0),
        sinus(1, 0, 0)
    ]

    fig, axs = plt.subplots(4)
    fig.suptitle('Exercitiul 6')
    x1s = xs[0](time)
    x2s = xs[1](time)
    x3s = xs[2](time)
    axs[0].plot(time, x1s)
    axs[1].plot(time, x2s)
    axs[2].plot(time, x3s)
    axs[3].plot(time, sinus(1, fs, 0)(time))
    plt.savefig('./labs/02/06.pdf')
    plt.show()

    # Observatie:
    # Sinusoida a) este similara cu cea initiala in vederea esantionarii
    # deoarece fs / 2 se afla la limita Nyquist.
    # Sinusoida b) este similara dar cu discrepante in anumite puncte.
    # Sinusoida c) este constanta cu x(t) = x[n] = 0, asemanator
    # curentului continuu (DC Signal).