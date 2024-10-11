import matplotlib.pyplot as plt
import sounddevice as sd
from math import prod
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def factorial(n):
    return prod(range(2, n + 1))

def taylor(x):
    x3 = x * x * x
    x5 = x * x * x3
    x7 = x * x * x5
    return x - (x3 / factorial(3)) + (x5 / factorial(5)) - (x7 / factorial(7))

def pade(alpha):
    up = alpha - (7 * alpha * alpha * alpha) / 60
    dw = 1 + (alpha * alpha) / 20
    return up / dw

if __name__ == '__main__':
    x = np.linspace(start = -np.pi / 2, stop = np.pi / 2, num = 10 ** 3)
    error = np.sin(x) - x
    fig, axs = plt.subplots(4)
    axs[0].set_title('sin(x)', fontstyle='italic')
    axs[0].plot(x, np.sin(x))
    axs[1].set_title('x', fontstyle='italic')
    axs[1].plot(x, x)
    axs[2].set_title('Error', fontstyle='italic')
    axs[2].plot(x, error)
    axs[3].set_yscale('log')
    axs[3].set_title('Log Error', fontstyle='italic')
    axs[3].plot(x, error)
    plt.savefig('./labs/02/08.pdf')
    plt.show()

    error = np.sin(x) - pade(x)
    fig, axs = plt.subplots(4)
    axs[0].set_title('sin(x)', fontstyle='italic')
    axs[0].plot(x, np.sin(x))
    axs[1].set_title('pade(x)', fontstyle='italic')
    axs[1].plot(x, pade(x))
    axs[2].set_title('Error', fontstyle='italic')
    axs[2].plot(x, error)
    axs[3].set_yscale('log')
    axs[3].set_title('Log Error', fontstyle='italic')
    axs[3].plot(x, error)
    plt.savefig('./labs/02/08p.pdf')
    plt.show()

    error = np.sin(x) - taylor(x)
    fig, axs = plt.subplots(4)
    axs[0].set_title('sin(x)', fontstyle='italic')
    axs[0].plot(x, np.sin(x))
    axs[1].set_title('taylor(x)', fontstyle='italic')
    axs[1].plot(x, taylor(x))
    axs[2].set_title('Error', fontstyle='italic')
    axs[2].plot(x, error)
    axs[3].set_yscale('log')
    axs[3].set_title('Log Error', fontstyle='italic')
    axs[3].plot(x, error)
    plt.savefig('./labs/02/08t.pdf')
    plt.show()

    plt.plot(x, np.sin(x) - x, marker='o', label='Identity Error', color='blue')
    plt.plot(x, np.sin(x) - pade(x), marker='o', label='Pade Error', color='orange')
    plt.plot(x, np.sin(x) - taylor(x), marker='o', label='Taylor Error', color='green')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./labs/02/08err.pdf')
    plt.show()