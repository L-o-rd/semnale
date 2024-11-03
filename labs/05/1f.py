import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    samples = samples - np.mean(samples)
    N = len(samples)
    X = np.fft.fft(samples)
    fq = np.fft.fftfreq(len(X), d = 3600)
    X = abs(X / N)[:N // 2]
    f = (1 / 3600) * np.linspace(0, N / 2, N // 2) / N
    indx = np.argsort(X)[-5:]
    freqs = (1 / 3600) * indx / N
    vals = X[indx]

    print('Top frecvente: ')
    for i in range(len(indx)):
        print('{0:.9f} Hz - {1:.2f} |X(w)|'.format(freqs[i], vals[i]))
        print('{0:.2f} days'.format((1 / freqs[i]) / 60 / 60 / 24))

    # 0.000011575 Hz reprezinta ~   1   zi
    # 0.000001656 Hz reprezinta ~   7 zile
    # 0.000000046 Hz reprezinta ~ 254 zile
    # 0.000000030 Hz reprezinta ~ 381 zile
    # 0.000000015 Hz reprezinta ~ 762 zile

    plt.stem(freqs, vals)
    plt.tight_layout()
    plt.savefig('./labs/05/1f.pdf')
    plt.show()