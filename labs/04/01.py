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
    x = sinus(1, 5, 0)

    Ns = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    tdfts = []
    tffts = []

    try:
        with open('./labs/04/tdfts.npy', 'rb') as f:
            tdfts = np.load(f)
    except:
        tdfts = []

        for N in Ns:
            ts = np.linspace(0, 1, N)
            y = x(ts)

            t1 = time.perf_counter()
            dft(y, N)
            t1 = time.perf_counter() - t1
            tdfts.append(t1)
        
        with open('./labs/04/tdfts.npy', 'wb') as f:
            np.save(f, tdfts)

    try:
        with open('./labs/04/tffts.npy', 'rb') as f:
            tffts = np.load(f)
    except:
        tffts = []

        for N in Ns:
            ts = np.linspace(0, 1, N)
            y = x(ts)

            t2 = time.perf_counter()
            np.fft.fft(y, N)
            t2 = time.perf_counter() - t2
            tffts.append(t2)
        
        with open('./labs/04/tffts.npy', 'wb') as f:
            np.save(f, tffts)

    plt.plot(Ns, tdfts, 'r')
    plt.plot(Ns, tffts, 'g')
    plt.xlabel('N')
    plt.ylabel('T (s)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./labs/04/01.pdf')
    plt.show()