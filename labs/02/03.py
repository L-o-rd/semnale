import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy as sp

def sinus(A, f0, phi):
    def x(t):
        return A * np.sin(2 * np.pi * f0 * t + phi)

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
    # a)

    f0 = 400        # Hz
    time = np.linspace(start = 0.0, stop = 1.0, num = 10 ** 3)
    nsamples = 44100 # samples
    ts = 1 / nsamples
    samples = np.arange(nsamples)
    data = np.sin(2 * np.pi * f0 * samples * ts)
    play(data, nsamples)

    # b)

    f0 = 800        # Hz
    time = np.linspace(start = 0.0, stop = 3.0, num = 10 ** 3)
    nsamples = 44100 # samples
    ts = 1 / nsamples
    samples = np.arange(nsamples)
    data = np.sin(2 * np.pi * f0 * samples * ts)
    play(data, nsamples)

    # c)

    f0 = 240
    data = 2 * (f0 * samples * ts - np.floor(f0 * samples * ts + 0.5))
    play(data, nsamples)
    sp.io.wavfile.write('./labs/02/03c.wav', rate = nsamples, data = data)
    rate, data = sp.io.wavfile.read('./labs/02/03c.wav')
    play(data, rate)

    # d)

    f0 = 300
    data = np.sign(np.sin(2 * np.pi * samples * ts * f0))
    play(data, nsamples)