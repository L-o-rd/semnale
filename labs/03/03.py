import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import numpy as np
import scipy as sp
import math

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def cosinus(A, f0, phi):
    def x(t):
        return A * np.cos(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    N = 100
    fs = 400
    time = np.linspace(start = 0.0, stop = 1.0, num = fs)

    f0 = 4
    x = sinus(1, f0, 0)
    y = cosinus(1 / 2, f0 * 30, np.pi / 3)
    z = sinus(2, f0 * 10, np.pi / 4)

    w = x(time) + y(time) + z(time)
    
    F = np.ones((N, N), dtype = complex)
    for l in range(N):
        for c in range(N):
            F[l, c] = math.e ** (-2j * math.pi * l * c * 1.0 / N)
    
    modx = np.matmul(F, w[:N])
    modx = np.abs(modx)
    fix, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(time, w)
    axs[0].set_xlabel('Timp (s)')
    axs[0].set_ylabel('w(t)')

    M = (N // 2) + (N % 2)
    axs[1].stem(np.arange(M) * fs / N, modx[:M])
    axs[1].set_xlabel('Frecventa (Hz)')
    axs[1].set_ylabel('|X(w)|')
    plt.savefig('./labs/03/03a.pdf')
    plt.tight_layout()
    plt.show()

    ci = np.argmax(modx[:M])
    cf = ci * fs / N
    F = np.ones((N, N), dtype = complex)
    for l in range(N):
        for c in range(N):
            F[l, c] = math.e ** (-2j * math.pi * l * c * 1.0 / N / cf)
    
    modx = np.matmul(F, w[:N])
    modx = np.abs(modx)
    fix, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(time, w)
    axs[0].set_xlabel('Timp (s)')
    axs[0].set_ylabel('w(t)')

    M = (N // 2) + (N % 2)
    axs[1].stem(np.arange(M) * fs / N, modx[:M])
    axs[1].set_xlabel('Frecventa (Hz)')
    axs[1].set_ylabel('|X(w)|')
    plt.savefig('./labs/03/03b.pdf')
    plt.tight_layout()
    plt.show()