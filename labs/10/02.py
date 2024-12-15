import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # (1) Liniar

    x = np.linspace(-1, 1, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    k = lambda x, y: x @ y.T
    c = k(x, x)
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 5)
    
    plt.figure(figsize = (10, 5))
    plt.title("Liniar Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02a.pdf')
    plt.show()

    # (2) Wiener

    x = np.linspace(0, 2, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    k = lambda x, y: np.minimum(x, y.T)
    c = k(x, x)
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 5)

    plt.figure(figsize = (10, 5))
    plt.title("Wiener Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02b.pdf')
    plt.show()

    # (3) Radial Basis Function

    x = np.linspace(-1, 1, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    alpha = 13
    k = lambda x, y: np.exp(-alpha * np.inner(x - y, x - y))
    c = np.array([[k(z, y) for y in x] for z in x])
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 5)

    plt.figure(figsize = (10, 5))
    plt.title("RBF Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02c.pdf')
    plt.show()

    # (4) Ornstein - Uhlenbeck

    x = np.linspace(0, 2, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    alpha = 13
    k = lambda x, y: np.exp(-alpha * np.abs(x - y)[0])
    c = np.array([[k(z, y) for y in x] for z in x])
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 3)

    plt.figure(figsize = (10, 5))
    plt.title("Ornstein - Uhlenbeck Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02d.pdf')
    plt.show()

    # (5) Periodic

    x = np.linspace(-1, 1, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    alpha = 0.5
    beta = 1.0
    k = lambda x, y: np.exp(-alpha * np.sin(beta * np.pi * (x - y)[0]) * np.sin(beta * np.pi * (x - y)[0]))
    c = np.array([[k(z, y) for y in x] for z in x])
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 5)

    plt.figure(figsize = (10, 5))
    plt.title("Periodic Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02e.pdf')
    plt.show()

    # (6) Simetric

    x = np.linspace(-1, 1, 10 ** 2).reshape(-1, 1)
    mu = np.zeros(x.shape)
    alpha = 13

    def k(x, y):
        global alpha
        min1 = np.abs(x - y)[0]
        min2 = np.abs(x + y)[0]
        mn = np.minimum(min1, min2)
        return np.exp(-alpha * mn * mn)

    c = np.array([[k(z, y) for y in x] for z in x])
    samples = np.random.multivariate_normal(mu.flatten(), c, size = 3)

    plt.figure(figsize = (10, 5))
    plt.title("Symmetric Process")
    plt.plot(x, samples.T)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.savefig('./labs/10/02f.pdf')
    plt.show()
