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
    orders = [2, 5, 9]
    rps = [1, 3, 5]
    rpcolor = ['red', 'orange', 'green']

    plt.figure(figsize=(15, 10))

    for i, order in enumerate(orders):
        bbut, abut = sp.signal.butter(order, Wn, btype='low')
        filtbut = sp.signal.filtfilt(bbut, abut, samples)
        
        plt.subplot(3, 2, i*2 + 1)
        plt.plot(samples, label='Original', marker='o')
        plt.plot(filtbut, label=f'Filtru Butterworth - (Order {order})', linestyle='--', color='orange')
        plt.title(f'Filtru Butterworth - Order {order}')
        plt.legend()
        plt.grid()
        
        for j, rp in enumerate(rps):
            bche, ache = sp.signal.cheby1(order, rp, Wn, btype='low')
            filtche = sp.signal.filtfilt(bche, ache, samples)
            
            plt.subplot(3, 2, i*2 + 2)
            plt.plot(samples, label='Original', marker='o', color='blue')
            plt.plot(filtche, label=f'Filtru Chebyshev (Order {order}, rp={rp} dB)', linestyle='--', color=rpcolor[j])
            plt.title(f'Filtru Chebyshev - Order {order}')
            plt.legend()
            plt.grid()

    plt.tight_layout()
    plt.savefig('./labs/06/04f.pdf')
    plt.show()

    # In continuare filtrul Butterworth
    # pare a fi cel mai centrat / simplu dar
    # definitor. Totusi, Chebyshev de ordin 5 - 9
    # si rp 1 dB pare a modela zgomotul
    # prezent in semnal mai bine ca Butterworth.