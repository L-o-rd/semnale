import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

    return x

def sawtooth(f0):
    def x(t):
        return 2 * (f0 * t - np.floor(f0 * t + 0.5))
    
    return x

def play(ys, sr):
    try:
        # Modify device accordingly
        # 2 should be the default output mapper

        sd.play(ys, samplerate = sr, device = 2)
        status = sd.wait()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
    if status:
        print('Error during playback: ' + str(status))

if __name__ == '__main__':
    f0 = 400        # Hz
    time = np.linspace(start = 0.0, stop = 1.0, num = 10 ** 3)
    nsamples = 44100 # samples
    ts = 1 / nsamples
    samples = np.arange(nsamples)
    data = np.sin(2 * np.pi * f0 * samples * ts)
    # print(data.shape)
    f0 = 800
    other = np.sin(2 * np.pi * f0 * samples * ts)
    # print(other.shape)
    data = np.concatenate([data, other])
    # print(data.shape)
    play(data, nsamples)

    # Observatie:
    # Prima parte suna exact ca o sinusoida de 400 Hz, a doua parte
    # cea de 800 Hz (sunet mai inalt) este introdusa printr-un
    # moment de tranzitie usor ridicat.
    # Diferenta se simte deoarece 800 Hz este cu o octava mai sus de 400 Hz.