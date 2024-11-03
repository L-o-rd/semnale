import matplotlib.pyplot as plt
import numpy as np

# Frecventa 0Hz in modulul FFT
# este cea mai semnificativa deci se observa
# prezenta unei componente continue.
# Ca sa scapam putem sa scadem media pentru
# a centra x(t).

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    samples = samples - np.mean(samples)
    N = len(samples)
    X = np.fft.fft(samples)
    fq = np.fft.fftfreq(len(samples), samples)
    X = abs(X / N)[:N // 2]
    f = (1 / 3600) * np.linspace(0, N / 2, N // 2) / N

    plt.plot(f, X)
    plt.tight_layout()
    plt.savefig('./labs/05/1e.pdf')
    plt.show()