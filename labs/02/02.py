import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    tstart = 0.0
    tstop = 1.0

    time = np.linspace(start = tstart, stop = tstop, num = 10 ** 2)
    f0 = 5.0 / (tstop - tstart)

    xs = [
        sinus(1, f0, 0),
        sinus(1, f0, np.pi / 2),
        sinus(1, f0, np.pi),
        sinus(1, f0, 3 * np.pi / 2)
    ]

    fig, axs = plt.subplots(4)
    fig.suptitle('Exercitiul 2')
    for i in range(len(xs)):
        axs[i].plot(time, xs[i](time))
        
    plt.savefig('./labs/02/02a.pdf')
    plt.show()

    zs = np.random.normal(loc = 0.0, scale = 1.0, size = (time.shape[0],))
    ys = xs[0](time)
    def snr(s):
        ysqr = np.linalg.norm(ys)
        ysqr *= ysqr
        zsqr = np.linalg.norm(zs)
        zsqr *= zsqr
        gammasqr = (ysqr / s) / zsqr
        return np.sqrt(gammasqr)

    snrs = [100, 10, 1, 0.1]
    fig, axs = plt.subplots(5)
    fig.suptitle('Exercitiul 2')
    axs[0].plot(time, ys)
    for i in range(len(snrs)):
        gamma = snr(snrs[i])
        noised = ys + gamma * zs
        axs[i + 1].plot(time, noised)
        
    plt.savefig('./labs/02/02b.pdf')
    plt.show()