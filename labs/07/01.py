from scipy import ndimage 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 100
    M = N

    X = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            X[i, j] = np.sin(2 * np.pi * i + 3 * np.pi * j)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01a.pdf')
    plt.show()

    X = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            X[i, j] = np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01b.pdf')
    plt.show()

    Y = np.zeros((N, M))
    Y[0, 5] = Y[0, N - 5] = 1
    X = np.real(np.fft.ifft2(Y))
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01c.pdf')
    plt.show()

    Y = np.zeros((N, M))
    Y[5, 0] = Y[N - 5, 0] = 1
    X = np.real(np.fft.ifft2(Y))
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01d.pdf')
    plt.show()

    Y = np.zeros((N, M))
    Y[5, 5] = Y[N - 5, N - 5] = 1
    X = np.real(np.fft.ifft2(Y))
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(abs(Y), cmap = 'gray')#(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01e.pdf')
    plt.show()

    X = np.zeros((N, M))
    mode = 0
    for i in range(N):
        for j in range(M):
            X[i, j] = (i + j) % 2

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(X, cmap = plt.cm.gray)
    Y = np.fft.fft2(X)
    axs[1].imshow(abs(Y), cmap = 'gray')#(10 * np.log10(abs(Y)))
    plt.tight_layout()
    plt.savefig('./labs/07/01f.pdf')
    plt.show()