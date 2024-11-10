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
    print(samples)
