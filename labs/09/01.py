import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    np.random.seed(3)
    N = 1000
    time = np.arange(N)

    trend = 0.0001 * time**2 - 0.005 * time + 5
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii
    #tidx = pd.date_range('2019-01-01', periods = N, freq = 'D')

    fig, axs = plt.subplots(4)
    axs[0].get_xaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks([])
    axs[2].get_xaxis().set_ticks([])

    axs[0].plot(y)
    axs[1].plot(trend)
    axs[2].plot(sezon)
    axs[3].plot(time, variatii)
    plt.tight_layout()
    plt.savefig('./labs/09/01.pdf')
    plt.show()