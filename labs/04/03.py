import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import math

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def fprint(N, F):
    for l in range(N):
        for c in range(N):
            print('({0.real:.2f} + {0.imag:.2f}j)'.format(F[l, c]), end = ' ')
        
        print()

def dft(x, N):
    F = np.ones((N, N), dtype = complex)
    for l in range(N):
        for c in range(N):
            F[l, c] = math.e ** (-2j * math.pi * l * c * 1 / N)

    return np.matmul(F, x)

if __name__ == '__main__':
    tms = np.linspace(0, 1, 8192)
    f0 = 5

    x = sinus(1, f0, 0)
    y = x(tms)

    fs = 7
    ts = 1 / fs
    fnyq = 11
    tnyq = 1 / fnyq

    samples = np.arange(fnyq)
    xn = x(samples * tnyq)

    fig, axs = plt.subplots(4)

    axs[0].plot(tms, y)

    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].plot(tms, y)
    axs[1].stem(samples * tnyq, xn, linefmt = 'r', basefmt = ' ')

    x = sinus(1, f0 + fs, 0)
    xn = x(samples * tnyq)
    y = x(tms)

    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].plot(tms, y)
    axs[2].stem(samples * tnyq, xn, linefmt = 'r', basefmt = ' ')

    x = sinus(1, f0 - fs, 0)
    xn = x(samples * tnyq)
    y = x(tms)

    axs[3].axhline(0, color='black', linestyle='--')
    axs[3].plot(tms, y)
    axs[3].stem(samples * tnyq, xn, linefmt = 'r', basefmt = ' ')
    
    plt.tight_layout()
    plt.savefig('./labs/04/03.pdf')
    plt.show()