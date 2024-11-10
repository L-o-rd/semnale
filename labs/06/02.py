import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

if __name__ == '__main__':
    N = 10
    px = np.random.randint(-10, 10, N + 1)
    qx = np.random.randint(-10, 10, N + 1)
    rx = np.zeros((2 * (N + 1),))
    for i in range(N + 1):
        for j in range(N + 1):
            pw = (i + j)
            cf = px[i] * qx[j]
            rx[pw] += cf

    px = np.append(px, np.zeros(2 * (N + 1) - len(px)))
    qx = np.append(qx, np.zeros(2 * (N + 1) - len(qx)))
    rft = np.fft.fft(px) * np.fft.fft(qx)
    rft = np.fft.ifft(rft)

    print('r(x)  = [ ', end = '')
    for rz in rx:
        print(f' {rz:1.0f} ', end = '')

    print(']')
    print('r\'(x) = [ ', end = '')
    for rz in rft:
        print(f' {rz.real:1.0f} ', end = '')

    print(']')
    print('r(x) =? r\'(x) ?', np.allclose(rx, rft))
    exit()