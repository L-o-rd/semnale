import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def boxcar(N):
    return np.ones(N)

def hann(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

if __name__ == '__main__':
    t = np.linspace(0.0, 2.0 / 10.0, 1000)
    x = sinus(1, 100, 0)
    y = x(t)

    Nw = 200
    fig, axs = plt.subplots(3)
    axs[0].plot(t[:Nw], y[:Nw])
    box = boxcar(Nw) # sp.signal.windows.boxcar(Nw)
    yft = y[:Nw] * box
    axs[1].plot(t[:Nw], yft)
    box = hann(Nw) # sp.signal.windows.hann(Nw)
    yft = y[:Nw] * box
    axs[2].plot(t[:Nw], yft)
    plt.tight_layout()
    plt.savefig('./labs/06/03.pdf')
    plt.show()
