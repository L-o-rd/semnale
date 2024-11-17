import scipy.datasets
from scipy import ndimage 
import numpy as np
import matplotlib.pyplot as plt

def snr(xs, zs, s):
    ysqr = np.linalg.norm(xs)
    ysqr *= ysqr
    zsqr = np.linalg.norm(zs)
    zsqr *= zsqr
    gammasqr = (ysqr / s) / zsqr
    return np.sqrt(gammasqr)

def ssnr(xs, ys):
    xsqr = np.linalg.norm(xs)
    xsqr *= xsqr

    ysqr = np.linalg.norm(ys - xs)
    ysqr *= ysqr
    return xsqr / ysqr

def isnr(snr, xs):
    xsqr = np.linalg.norm(xs)
    xsqr *= xsqr
    return np.sqrt(xsqr / snr) + X

if __name__ == '__main__':
    X = scipy.datasets.face(gray = True)
    Z = np.random.randint(-200, high = 201, size = X.shape)
    gamma = snr(X, Z, 0.1)
    Noisy = X + gamma * Z

    Y = np.fft.fft2(X)
    freqx = np.fft.fftfreq(X.shape[1])
    freqy = np.fft.fftfreq(X.shape[0])
    freq_db = 20 * np.log10(abs(Y))
    freq_cutoff = 110
    Y_cutoff = Y.copy()
    # Y_cutoff[(freqx + freqy) > freq_cutoff] = 0
    ifrqcuty = int(0.49 * Y_cutoff.shape[0])
    ifrqcutx = int(0.49 * Y_cutoff.shape[1])
    halfy = Y_cutoff.shape[0] // 2
    halfx = Y_cutoff.shape[1] // 2
    Y_cutoff[halfy - ifrqcuty : halfy + ifrqcuty, halfx - ifrqcutx : halfx + ifrqcutx] = 0
    #plt.imshow(20 * np.log10(abs(Y_cutoff)))
    #plt.show()
    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)

    print(ssnr(X, X_cutoff))
    SNR = 31
    YS = isnr(SNR, X)
    fig, axs = plt.subplots(2, 2, figsize = (10, 6))
    axs[0, 0].imshow(X, cmap = 'gray')
    axs[0, 1].imshow(freq_db)
    axs[1, 0].imshow(YS, cmap = 'gray')
    axs[1, 1].imshow(X_cutoff, cmap=plt.cm.gray)
    plt.savefig('./labs/07/02.pdf')
    plt.show()