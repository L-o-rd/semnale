import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

if __name__ == '__main__':
    samples = np.genfromtxt('./labs/05/Train.csv', delimiter=',')
    samples = samples[1:, 2]
    N = len(samples)
    ds = 3 * 24 * 60 * 60 # 3 zile -> secunde
    dh = 3 * 24 # 3 zile -> ore
    samples = samples[:dh]
    
    fs = 1 / 3600
    fnyq = fs / 2
    
    # Orice frecventa mai mare de fnyq
    # nu poate fi reprezentata corect,
    # din cauza alierii deoarece fs = 1 / 3600
    # deci putem alege o frecventa mai mica de fnyq
    # pentru a vedea ciclul zilnic am putea alege
    # fc = 1 / 24 h = 1 / (24 * 3600) = 1 / 86400 = 0.00001157 Hz
    # deci putem alege fc sa fie putin mai mare pentru a
    # pastra ciclul zilnic => fc = 0.00002 Hz
    # fnorm = fc / fnyq = (2 * 10^-5) / (1 / (2 * 3600))
    # fnorm = 0.144

    fc = 2 / (10 ** 5)
    fnorm = fc / fnyq
    Wn = fnorm
    order = 5

    bbut, abut = sp.signal.butter(order, Wn, btype='low')

    rp = 5
    bche, ache = sp.signal.cheby1(order, rp, Wn, btype='low')

    filtbut = sp.signal.filtfilt(bbut, abut, samples)
    filtche = sp.signal.filtfilt(bche, ache, samples)

    plt.figure(figsize=(12, 6))
    plt.plot(samples, label='Original', marker='o')
    plt.plot(filtbut, label='Butterworth', linestyle='--', color='orange')
    plt.plot(filtche, label=f'Chebyshev (rp={rp} dB)', linestyle='--', color='green')
    plt.legend()
    plt.savefig('./labs/06/04e.pdf')
    plt.grid()
    plt.show()

    # In cazul acesta cu rp = 5 dB Chebyshev pare a
    # fi mai slab ca Butterworth, deoarece
    # Butterworth filtreaza samples intr-un
    # mod mai smooth si are frecventele
    # mai apropiate.

