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

    w, hbut = sp.signal.freqz(bbut, abut, worN=8000)
    w, hche = sp.signal.freqz(bche, ache, worN=8000)

    plt.figure(figsize=(12, 6))
    plt.plot(w * fnyq / np.pi, 20 * np.log10(abs(hbut)), label='Filtrul Butterworth')
    plt.plot(w * fnyq / np.pi, 20 * np.log10(abs(hche)), label=f'Filtrul Chebyshev (rp={rp} dB)')
    plt.xlabel('Frecventa (Hz)')
    plt.ylabel('Amplitudine (dB)')
    plt.legend()
    plt.savefig('./labs/06/04d.pdf')
    plt.grid()
    plt.show()

