import matplotlib.pyplot as plt
import numpy as np

'''
    1. Fie semnalele continue:
        x(t) = cos(520πt + π/3), 
        y(t) = cos(280πt − π/3),
        z(t) = cos(120πt + π/3).
'''

def x(t):
    return np.cos(520.0 * np.pi * t + np.pi / 3.0)

def y(t):
    return np.cos(280.0 * np.pi * t - np.pi / 3.0)

def z(t):
    return np.cos(120.0 * np.pi * t + np.pi / 3.0)

if __name__ == '__main__':
    # a)
    xs = np.arange(start = 0.0, step = 0.0001, stop = 0.1)
    plt.title('Exercitiul 1a')
    plt.plot(xs, xs)
    plt.savefig('./labs/01/01a.pdf')
    plt.show()

    # b)
    fig, axs = plt.subplots(3)
    fig.suptitle('Exercitiul 1b')
    axs[0].plot(xs, x(xs))
    axs[1].plot(xs, y(xs))
    axs[2].plot(xs, z(xs))
    plt.savefig('./labs/01/01b.pdf')
    plt.show()

    # c)
    fs = 200.0                      # Hz
    ts = 1.0 / fs                   # s
    nsamples = fs * 0.1
    samples = np.arange(nsamples)

    fig, axs = plt.subplots(3)
    fig.suptitle('Exercitiul 1c')
    axs[0].plot(xs, x(xs))
    axs[0].stem(samples * ts, x(samples * ts))
    axs[1].plot(xs, y(xs))
    axs[1].stem(samples * ts, y(samples * ts))
    axs[2].plot(xs, z(xs))
    axs[2].stem(samples * ts, z(samples * ts))
    plt.savefig('./labs/01/01c.pdf')
    plt.show()
    