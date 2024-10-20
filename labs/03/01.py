import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    N = 8
    F = np.ones((N, N), dtype = complex)
    for l in range(N):
        for c in range(N):
            F[l, c] = math.e ** (-2j * math.pi * l * c * 1 / N)

    fig, axs = plt.subplots(N)
    fig.suptitle('Exercitiul 1')
    for i in range(N):
        axs[i].plot(np.arange(N), F[i].real)
        axs[i].plot(np.arange(N), F[i].imag, '--')

    plt.savefig('./labs/03/01.pdf')
    plt.show()

    FH = np.conjugate(F)
    FH = FH.transpose()
    UN = np.matmul(F, FH)

    unitary = np.allclose(UN / N, np.identity(N, dtype = complex))
    if unitary: print('F este unitara.')
    else: print('F nu este unitara.')

    

    