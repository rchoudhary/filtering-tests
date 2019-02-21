import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import *

f0 = 20
fs = 1000
T = 4  # observation length (s)
t = np.linspace(0, T, fs * T)

# generate cosine wave
x = np.cos(2 * np.pi * f0 * t)
# apply windowing function

window = signal.windows.blackmanharris(len(x))
x_windowed = x * window

# zero padding
# np.append(x_windowed, np.zeros(10000))

X = fftshift(fft(x_windowed))
X_db = 20 * np.log10(np.abs(X))
f = np.fft.fftshift(np.fft.fftfreq(len(x)) * fs)

fig, ax = plt.subplots()
ax.set_ylim(-120, 85)
ax.set_xlim(-50, 50)
y_major_ticks = np.arange(-120, 85, 20)
ax.set_yticks(y_major_ticks)
x_minor_ticks = np.arange(-50, 50, 10)
ax.set_xticks(x_minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.5)
plt.axhline(0, linewidth=1, color='black')
plt.axvline(0, linewidth=1, color='black')
ax.plot(f, X_db)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
plt.title('FFT of a Sinusoid')
plt.show()