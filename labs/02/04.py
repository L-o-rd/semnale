import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def sawtooth(f0):
    def x(t):
        return 2 * (f0 * t - np.floor(f0 * t + 0.5))
    
    return x

if __name__ == '__main__':
    time = np.linspace(start = 0.0, stop = 1.0, num = 10 ** 3)
    f0 = 5.0 / (1.0 - 0.0)
    x = sinus(1, f0, 0)
    y = sawtooth(f0)

    fig, axs = plt.subplots(3)
    fig.suptitle('Exercitiul 4')
    xs = x(time)
    ys = y(time)
    zs = xs + ys
    axs[0].plot(time, xs)
    axs[1].plot(time, ys)
    axs[2].plot(time, zs)
    plt.savefig('./labs/02/04.pdf')
    plt.show()