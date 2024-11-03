import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    samples = samples
    N = len(samples)
    X = np.fft.fft(samples)
    fq = np.fft.fftfreq(len(samples), samples)
    X = abs(X / N)[:N // 2]
    fs = 1 / 3600
    f = fs * np.linspace(0, N / 2, N // 2) / N
    nsample = 1368  # 2 months
    nsamples = int(np.ceil(2629743.83 * fs))
    month_samples = samples[nsample : (nsample + nsamples)]

    plt.plot(np.arange(nsamples), month_samples)
    plt.tight_layout()
    plt.savefig('./labs/05/1g.pdf')
    plt.show()