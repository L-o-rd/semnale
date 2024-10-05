import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # a)
    f0 = 400        # Hz
    nsamples = 1600 # samples
    fs = nsamples / (10 / f0)
    ts = 1 / fs
    
    xs = np.arange(start = 0.0, stop = 10 / f0, step = 0.00001)
    samples = np.arange(nsamples)
    ys = np.sin(xs * 2 * np.pi * f0)
    plt.title('Exercitiul 2a')
    plt.plot(xs, ys)
    plt.stem(samples * ts, np.sin(2 * np.pi * f0 * samples * ts))
    plt.savefig('./labs/01/02a.pdf')
    plt.show()

    # b)
    f0 = 800                        # Hz
    T = 3                           # s
    xs = np.arange(T, step = 0.001)
    ys = np.sin(xs * 2 * np.pi * f0)
    xssub = np.arange(10 / f0, step = 0.00001)
    yssub = np.sin(xssub * 2 * np.pi * f0)
    plt.title('Exercitiul 2b')
    plt.plot(xssub, yssub)
    plt.savefig('./labs/01/02b.pdf')
    plt.show()

    # c)
    f0 = 240
    xs = np.arange(5 / f0, step = 0.0001)
    ys = 2 * (f0 * xs - np.floor(f0 * xs + 0.5))
    plt.title('Exercitiul 2c')
    plt.plot(xs, ys)
    plt.savefig('./labs/01/02c.pdf')
    plt.show()

    # d)
    f0 = 300
    xs = np.arange(5 / f0, step = 0.0001)
    ys = np.sign(np.sin(2 * np.pi * xs * f0))
    plt.title('Exercitiul 2d')
    plt.plot(xs, ys)
    plt.savefig('./labs/01/02d.pdf')
    plt.show()

    # e)
    xs = np.random.rand(128, 128)
    plt.title('Exercitiul 2e')
    plt.imshow(xs)
    plt.savefig('./labs/01/02e.pdf')
    plt.show()

    # f) Basic XOR Texture 128 x 128
    xs = np.zeros((128, 128))
    for x in range(0, 128):
        for y in range(0, 128):
            xs[x, y] = x ^ y
    
    plt.title('Exercitiul 2f')
    plt.imshow(xs)
    plt.savefig('./labs/01/02f.pdf')
    plt.show()
    
