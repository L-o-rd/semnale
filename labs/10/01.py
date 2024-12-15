import matplotlib.pyplot as plt
import numpy as np

def gauss(n, mu = 0.0, vr = 1.0):
    return np.random.normal(loc = mu, scale = np.sqrt(vr), size = n)

def dgauss(x, mu = 0.0, vr = 1.0):
    return (1 / (np.sqrt(2 * np.pi * vr))) * np.exp(-((x - mu) ** 2) / (2 * vr))

def bell(n, mu = 0.0, vr = 1.0, hist = False):
    dev = np.sqrt(vr)
    x = np.linspace(-5, 5, n)
    return (x, dgauss(x, mu, vr))

if __name__ == '__main__':
    plt.figure(figsize = (10, 5))
    x, y = bell(10 ** 3, 0, 1.0)
    plt.plot(x, y, 'r-', lw = 2, label = 'μ = 0, σ * σ = 1.0')
    
    x, y = bell(10 ** 3, 0, 0.2)
    plt.plot(x, y, 'b-', lw = 2, label = 'μ = 0, σ * σ = 0.2')

    x, y = bell(10 ** 3, 0, 5.0)
    plt.plot(x, y, 'y-', lw = 2, label = 'μ = 0, σ * σ = 5.0')

    x, y = bell(10 ** 3, -2, 0.5)
    plt.plot(x, y, 'g-', lw = 2, label = 'μ = -2, σ * σ = 0.5')

    plt.title('Gaussian Distributions')
    plt.xticks(np.arange(-5, 6, 1))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlabel("X")
    plt.ylabel("φ(X)")
    plt.tight_layout()
    plt.legend()
    plt.savefig('./labs/10/01a.pdf')
    plt.show()

    plt.figure(figsize = (10, 5))
    x, y = bell(10 ** 3, 0, 1.0)
    plt.plot(x, y, 'r-', lw = 2, label = 'μ = 0, σ * σ = 1.0')
    samples = gauss(10 ** 3)
    plt.hist(samples, bins = 30, density = True, alpha = 0.6, color = 'b')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./labs/10/01b.pdf')
    plt.show()

    mean = [0, 0]
    cmat = [[1, 3 / 5],
            [3 / 5, 2]]
    eval, evec = np.linalg.eig(cmat)
    Amat = np.diag(eval)
    Aroot = np.diag(np.sqrt(eval))
    Umat = evec
    
    nvec = np.random.randn(10 ** 3, 2)
    xx = np.array(mean).reshape((2, 1)) + (Umat @ Aroot @ nvec.T)
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(projection='3d')
    samples = np.random.multivariate_normal(mean, cmat, 10 ** 3)

    ax.scatter(samples[:, 0], samples[:, 1], zs = 0, zdir = 'z', 
                alpha = 0.75, c = 'black', s = [8] * len(samples[:, 1]))
    
    samples = xx.T
    ax.scatter(samples[:, 0], samples[:, 1], zs = 0, zdir = 'z', 
                alpha = 0.35, c = 'blue', s = [8] * len(samples[:, 1]))

    x = np.linspace(-4, 4, 10 ** 3)
    y = np.linspace(-4, 4, 10 ** 3)
    z_x = (1 / (np.sqrt(2 * np.pi * cmat[0][0]))) * np.exp(-((x - mean[0]) ** 2) / (2 * cmat[0][0]))
    z_y = (1 / (np.sqrt(2 * np.pi * cmat[1][1]))) * np.exp(-((y - mean[1]) ** 2) / (2 * cmat[1][1]))

    ax.plot(x, z_x, color = 'b', zs =  4, zdir='y', lw = 2)
    ax.plot(y, z_y, color = 'r', zs = -4, zdir='x', lw = 2)

    theta = np.linspace(0, 2 * np.pi, 201)
    x = 4 * np.cos(theta)
    z = 4 * np.sin(theta)
    ax.plot(x, z, color = 'g')
    plt.savefig('./labs/10/01c.pdf')
    plt.tight_layout()
    plt.show()
