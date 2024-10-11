import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    time = np.linspace(start = 0.0, stop = 5 / 1000, num = 10 ** 4)
    f0 = 1000

    dcf = 4
    x = sinus(1, f0, 0)
    ys = x(time)
    dc = ys[::dcf]
    fig, axs = plt.subplots(2)
    axs[0].plot(time, ys)
    axs[1].stem(time[::dcf], dc)
    plt.savefig('./labs/02/07a.pdf')
    plt.show()

    dc = ys[1::dcf]
    fig, axs = plt.subplots(2)
    axs[0].plot(time, ys)
    axs[1].stem(time[1::dcf], dc)
    plt.savefig('./labs/02/07b.pdf')
    plt.show()

    # Observatie:
    # In acest caz deoarece noua frecventa va fi 250 Hz
    # semnalul este inca identificabil si seamana destul de mult
    # datorita interpolarii ce are loc.

    time = np.linspace(start = 0.0, stop = 10 / 400, num = 10 ** 3)
    fs = 1000
    ts = 1 / fs
    f0 = 400
    nsamples = (10 * fs) / 100
    samples = np.arange(nsamples)

    dcf = 4
    x = sinus(1, f0, 0)
    ys = x(time)
    zs = x(samples * ts)
    dc = zs[::dcf]
    tss = samples * ts
    fig, axs = plt.subplots(3)
    axs[0].plot(time, ys)
    axs[1].stem(tss, zs)
    axs[2].stem(tss[::dcf], dc)
    plt.savefig('./labs/02/07a2.pdf')
    plt.show()

    dc = zs[1::dcf]
    fig, axs = plt.subplots(3)
    axs[0].plot(time, ys)
    axs[1].stem(tss, zs)
    axs[2].stem(tss[1::dcf], dc)
    plt.savefig('./labs/02/07b2.pdf')
    plt.show()

    # Observatie:
    # In cazul acesta cu o frecventa de 400 Hz
    # si un fs de 1000 Hz, avem frecventa Nyquist = 500 Hz
    # deci semnalul esantionat va fi identificabil, dar
    # dupa decimare fs = 250 Hz si fn = 125 Hz deci
    # semnalul initial nu va mai fi identificabil (va avea loc aliasing)
    # Incepand cu al doilea element, se observa ca semnalul decimat
    # arata diferit de cel incepand de la primul element, deci
    # este neidentificabil.