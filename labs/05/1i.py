import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    N = len(samples)
    X = np.fft.fft(samples)
    fq = np.fft.fftfreq(len(X), d = 3600)
    X = abs(X / N)[:N // 2]
    cutoffi = 49 * N // 100

    fig, axs = plt.subplots(1, 3, figsize = (14, 6))
    axs[0].plot(np.arange(N), samples)
    axs[0].set_title('Before')

    filtered = np.fft.fft(samples)
    filtered[(N // 2 - cutoffi) : (N // 2 + cutoffi + 1)] = 0
    lfiltered = np.fft.ifft(filtered).real

    axs[1].plot(np.arange(N), lfiltered)
    axs[1].set_title('Mild')

    cutofffq = np.max(fq) / 500
    filtered = np.fft.fft(samples)
    filtered[np.abs(fq) >= cutofffq] = 0
    lfiltered = np.fft.ifft(filtered).real

    axs[2].plot(np.arange(N), lfiltered)
    axs[2].set_title('Drastic')
    plt.tight_layout()
    plt.savefig('./labs/05/1i.pdf')
    plt.show()