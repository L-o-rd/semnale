import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import numpy as np
import scipy as sp
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

def cmap(x, y):
    dst = np.sqrt(np.square(x) + np.square(y))
    norm = (dst - dst.min()) / (dst.max() - dst.min())
    return norm

if __name__ == '__main__':
    # Un semnal sinusoidal de frecventa fundamentala 7 Hz
    # Esantionat la o frecventa de 1000 Hz

    nsamples = 1000
    time = np.linspace(0.0, 1.0, nsamples)
    f0 = 7
    x = sinus(1, f0, np.pi / 2)
    fst = 5 * f0
    fs = f0 * fst
    ts = 1 / fs
    samples = np.arange(nsamples)
    #xn = x(samples * ts)
    xn = x(time)
    #yn = xn * (np.e ** (-2j * np.pi * 1 * (samples * ts)))
    yn = xn * (np.e ** (-2j * np.pi * time))
    xsample = 620

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Exercitiul 2a')
    axs[0].set_xlabel('Timp (esantioane)')
    axs[0].set_ylabel('Amplitudine')
    axs[0].plot(time, xn)
    #axs[0].stem(samples[nsamples >> 1], x(samples[nsamples >> 1] * ts), linefmt='r')
    axs[0].stem(time[xsample], xn[xsample], linefmt='r')
    axs[0].axhline(0, color='black', linestyle='--')
    
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].axvline(0, color='black', linestyle='--')
    axs[1].set_xlabel('Real')
    axs[1].set_ylabel('Imaginar')
    axs[1].scatter(yn.real, yn.imag, c = cmap(yn.real, yn.imag))
    # val = x(samples[nsamples >> 1] * ts) * (np.e ** (-2j * np.pi * 1 * (samples[nsamples >> 1] * ts)))
    val = yn[xsample]
    vals = np.linspace(min(val.real, 0.0), max(val.real, 0.0), 100)
    yvals = [(val.imag / val.real) * x for x in vals]
    axs[1].plot(vals, yvals, color = 'r')
    axs[1].stem(val.real, val.imag, basefmt=" ", linefmt="r-")
    plt.savefig('./labs/03/02a.pdf')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)
    for i in range(len(axs)):
        for j in range(len(axs[0])):
            axs[i, j].axhline(0, color='black', linestyle='--')
            axs[i, j].axvline(0, color='black', linestyle='--')
            axs[i, j].set_xlim([-1, 1])
            axs[i, j].set_ylim([-1, 1])

    fig.suptitle('Exercitiul 2b')
    axs[0, 0].set_xlabel('Real')
    axs[0, 0].set_ylabel('Imaginar')
    #axs[0, 0].plot(yn.real, yn.imag)
    axs[0, 0].scatter(yn.real, yn.imag, c = cmap(yn.real, yn.imag))

    #yn = xn * (np.e ** (-2j * np.pi * 2 * (samples * ts)))
    yn = xn * (np.e ** (-2j * np.pi * 2 * time))
    fig.suptitle('Exercitiul 2b')
    axs[0, 1].set_xlabel('Real')
    axs[0, 1].set_ylabel('Imaginar')
    #axs[0, 1].plot(yn.real, yn.imag)
    axs[0, 1].scatter(yn.real, yn.imag, c = cmap(yn.real, yn.imag))

    #yn = xn * (np.e ** (-2j * np.pi * 5 * (samples * ts)))
    yn = xn * (np.e ** (-2j * np.pi * 5 * time))
    fig.suptitle('Exercitiul 2b')
    axs[1, 0].set_xlabel('Real')
    axs[1, 0].set_ylabel('Imaginar')
    #axs[1, 0].plot(yn.real, yn.imag)
    axs[1, 0].scatter(yn.real, yn.imag, c = cmap(yn.real, yn.imag))
    
    #yn = xn * (np.e ** (-2j * np.pi * 7 * (samples * ts)))
    yn = xn * (np.e ** (-2j * np.pi * f0 * time))
    fig.suptitle('Exercitiul 2b')
    axs[1, 1].set_xlabel('Real')
    axs[1, 1].set_ylabel('Imaginar')
    #axs[1, 1].plot(yn.real, yn.imag)
    axs[1, 1].scatter(yn.real, yn.imag, c = cmap(yn.real, yn.imag))
    plt.savefig('./labs/03/02b.pdf')
    plt.tight_layout()
    plt.show()