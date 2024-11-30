import numpy as np
import matplotlib.pyplot as plt

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def arpred(y, s, p):
    n = len(y)
    m = n - p
    Y = np.zeros((m, p))
    yy = np.zeros((m, 1))
    for i in range(m):
        for j in range(p):
            Y[i, j] = y[s - i - (j + 1)]
        
        yy[i] = y[s - i]

    xstar = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Y), Y)), np.transpose(Y)), yy)
    yhat = np.matmul(np.transpose(xstar), yy[:p])
    return yhat

def mapred(y, p):
    n = len(y)
    errors = np.random.normal(0, 1, size = n)
    X = np.zeros((n - p, p))
    for t in range(n - 1, p - 1, -1):
        for j in range(p):
            X[n - 1 - t, j] = errors[t - (j + 1)]

    yhat = (y[p:][::-1] - errors[p:][::-1]) - np.mean(y)
    theta = (np.linalg.inv(X.T @ X) @ X.T) @ yhat
    return (np.mean(y) + np.random.normal(0, 1) + np.dot(theta, errors[-p:][::-1]))

if __name__ == '__main__':
    np.random.seed(1)
    N = 1000
    time = np.arange(N)

    trend = 0.0001 * time ** 2 - 0.005 * time + 5
    trend /= np.max(trend)
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii

    p = 50
    steps = 150
    ys = y.copy()
    for i in range(steps):
        yhat = mapred(ys, p)
        ys = np.append(ys, yhat)

    plt.figure(figsize = (12, 6))
    plt.plot(time, y, label = "Original")
    plt.plot(range(N, N + steps), ys[-steps:], label = "Predictii MA", linestyle = "dashed")
    plt.legend()
    plt.savefig('./labs/09/03.pdf')
    plt.show()