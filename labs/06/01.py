import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

if __name__ == '__main__':
    N = 100
    xn = np.random.rand(N)
    
    fig, axs = plt.subplots(4, figsize=(10, 6))
    axs[0].plot(xn, marker = 'o')
    axs[0].grid(True)
    xn = np.convolve(xn, xn, 'full')
    axs[1].plot(xn, marker = 'o')
    axs[1].grid(True)
    xn = np.convolve(xn, xn, 'full')
    axs[2].plot(xn, marker = 'o')
    axs[2].grid(True)
    xn = np.convolve(xn, xn, 'full')
    axs[3].plot(xn, marker = 'o')
    axs[3].grid(True)
    plt.savefig('./labs/06/01.pdf')
    plt.show()