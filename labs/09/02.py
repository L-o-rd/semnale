import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def expavg(x, alpha):
    s = np.zeros(x.shape[0])
    s[0] = x[0]
    for i in range(1, len(s)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]

    return s

def expval(s, x):
    sum = 0
    for i in range(len(s) - 1):
        sum += np.power((s[i] - x[i + 1]), 2)

    return sum

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

    alpha = 0.83
    s1 = expavg(y, alpha)
    
    bval = float('inf')
    balpha = 0
    alpha = np.random.random()
    dalpha = 0.1
    ts = 0

    while ts < 500:
        val0 = expval(expavg(y, alpha), y)
        val1 = expval(expavg(y, alpha + dalpha), y)
        # print(f'a = {alpha}, v = {val0}')
        if val0 < val1:
            alpha -= dalpha
        else:
            alpha += dalpha

        if val0 < bval:
            balpha = alpha
            bval = val0

        ts += 1
        if ts % 100 == 0:
            dalpha *= 0.1

    s2 = expavg(y, balpha)
    fig, axs = plt.subplots(3)
    axs[0].plot(y, label = f'Original')
    axs[0].legend()
    axs[1].plot(s1, label = f'Alpha = 0.83')
    axs[1].legend()
    axs[2].plot(s2, label = f'Alpha = {balpha:.2f}')
    axs[2].legend()
    plt.tight_layout()
    plt.savefig('./labs/09/02.pdf')
    plt.show()