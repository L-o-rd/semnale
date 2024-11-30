from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

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
    # sezon = 0.5 * np.sin(2 * np.pi * 4 * time) + 0.25 * np.sin(3 * np.pi * 10 * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii

    p, q = 2, 3
    steps = 150
    ys = y.copy()
    for i in range(steps):
        yhat_arma = arpred(ys, N + i - 1, p) + mapred(ys, q) - np.mean(y)
        ys = np.append(ys, yhat_arma)

    fig, axs = plt.subplots(3, figsize = (12, 6))
    axs[0].plot(time, y, label = "Original")
    axs[0].axvline(N, color = 'gray', linestyle = '--', label = 'Predictii')
    axs[0].plot(range(N, N + steps), ys[-steps:], label = f"Predictii ARMA (p = {p}, q = {q})", linestyle = "dashed")
    axs[0].legend(loc = 'upper left')

    prct = int(0.99 * N)
    rest = N - prct
    against = y[prct:]
    bmse = float('inf')
    bp, bq = 19, 4 # None, None (pentru a recalcula)
    if bp == None and bq == None:
        for p in range(2, 20):
            for q in range(2, 20):
                train = y[:prct].copy()
                for i in range(rest):
                    yhat_arma = arpred(train, prct + i - 1, p) + mapred(train, q) - np.mean(train)
                    train = np.append(train, yhat_arma)
                
                mse = (np.square(train[-rest:] - against)).mean()
                if mse < bmse:
                    bmse = mse
                    bp, bq = p, q

    p, q = bp, bq
    steps = 150
    ys = y.copy()
    for i in range(steps):
        yhat_arma = arpred(ys, N + i - 1, p) + mapred(ys, q) - np.mean(y)
        ys = np.append(ys, yhat_arma)

    axs[1].plot(time, y, label = "Original")
    axs[1].axvline(N, color = 'gray', linestyle = '--', label = 'Predictii')
    axs[1].plot(range(N, N + steps), ys[-steps:], label = f"Predictii ARMA optim (p = {bp}, q = {bq})", linestyle = "dashed")
    axs[1].legend(loc = 'upper left')

    p, d, q = 3, 0, 5
    model = ARIMA(y, order = (p, d, q))
    model = model.fit()
    steps = 150
    preds = model.forecast(steps = steps)

    axs[2].plot(time, y, label = "Original")
    axs[2].axvline(N, color = 'gray', linestyle = '--', label = 'Predictii')
    axs[2].plot(range(N, N + steps), preds, label = f"Predictii ARIMA (p = {p}, d = {d}, q = {q})", linestyle = "dashed")
    axs[2].legend(loc = 'upper left')
    plt.savefig('./labs/09/04.pdf')
    plt.show()