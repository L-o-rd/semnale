import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def arpred(y, s, m, p):
    Y = np.zeros((m, p))
    yy = np.zeros((m, 1))
    for i in range(m):
        for j in range(p):
            Y[i, j] = y[s - i - (j + 1)]
        
        yy[i] = y[s - i]

    xstar = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Y), Y)), np.transpose(Y)), yy)
    yhat = np.matmul(np.transpose(xstar), yy[:p])
    return yhat

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

    bstp = None
    bstm = None
    babs = float('inf')

    ytruth = y[N - 1]
    ys = y.copy()[:N - 1]
    for p in range(1, 14):
        for m in range(p, N // 2 + 1):
            yhat = arpred(ys, N - 2, m, p)
            if np.abs(yhat - ytruth) < babs:
                babs = np.abs(yhat - ytruth)
                bstp = p
                bstm = m

    print(f'Best p: {bstp}, Best m: {bstm}')

    steps = 150
    ys = y.copy()
    for i in range(steps):
        yhat = arpred(ys, N + i - 1, bstm, bstp)
        ys = np.append(ys, yhat)

    plt.figure(figsize = (12, 6))
    plt.plot(time, y, label = "Original")
    plt.plot(range(N, N + steps), ys[-steps:], label = "Predictii AR", linestyle = "dashed")
    plt.legend()
    plt.savefig('./labs/08/01d.pdf')
    plt.show()
