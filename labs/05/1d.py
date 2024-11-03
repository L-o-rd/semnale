import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    N = len(samples)
    X = np.fft.fft(samples)
    fq = np.fft.fftfreq(len(samples), samples)
    X = abs(X / N)[:N // 2]
    f = (1 / 3600) * np.linspace(0, N / 2, N // 2) / N

    plt.plot(f, X)
    plt.tight_layout()
    plt.savefig('./labs/05/1d.pdf')
    plt.show()