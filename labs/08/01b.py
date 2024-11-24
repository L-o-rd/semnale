import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def acor(y, lag):
    yl = y[:len(y) - lag]
    yl = np.concatenate([yl, [0] * lag])
    return np.dot(y, yl)

if __name__ == '__main__':
    np.random.seed(1)
    N = 1000
    time = np.arange(N)

    trend = 0.0001 * time**2 - 0.005 * time + 5
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii

    acors = np.zeros(N)
    for i in range(N):
        acors[i] = acor(y, i + 1)

    autocorr = np.convolve(y, np.flip(y), mode = 'full')
    fig, axs = plt.subplots(3)
    axs[1].plot(autocorr / np.dot(y, y), label = 'Autocorelatie')
    axs[0].plot(np.correlate(y, y, mode = 'full') / (np.dot(y, y)), label = 'Autocorelatie (np)')
    axs[2].plot(acors / np.dot(y, y))
    plt.tight_layout()
    axs[1].legend()
    axs[0].legend()
    plt.savefig('./labs/08/01b.pdf')
    plt.show()