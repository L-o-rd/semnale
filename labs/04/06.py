import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import wavfile

sample_rate, samples = wavfile.read('./labs/04/sound.wav')

segment_size = int(0.01 * len(samples))
overlap = int(0.5 * segment_size)

num_segments = (len(samples) - overlap) // (segment_size - overlap)
spectrogram_matrix = np.zeros((segment_size // 2, num_segments))

for i in range(num_segments):
    start = i * (segment_size - overlap)
    end = start + segment_size
    segment = samples[start:end]
    
    fft_values = np.abs(fft(segment)[:segment_size // 2])
    spectrogram_matrix[:, i] = fft_values

times = np.linspace(0, len(samples) / sample_rate, num_segments)
frequencies = np.fft.fftfreq(segment_size, 1 / sample_rate)[:segment_size // 2]

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_matrix), shading='gouraud')
plt.colorbar(label='Intensitate (dB)')
plt.ylabel('Frecventa (Hz)')
plt.xlabel('Timp (s)')
plt.title('Spectograma')
plt.ylim(0, 7000)
plt.savefig('./labs/04/06.pdf')
plt.show()

