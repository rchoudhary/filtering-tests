import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from scipy import signal

f0 = 100  # square wave freq (Hz)
fs = 2000
T = 1  # observation length (s)
Z = T * 50  # zero stretch length (s)
t = np.linspace(0, T, fs * T)
x = 0.5 + 0.5 * signal.square(2 * np.pi * f0 * t)  # square wave with frequency f0

# low pass filter
order = 6
cutoff_low = 100  # cutoff frequency (Hz)
nyq = 0.5 * fs
norm_low = (cutoff_low - 20) / nyq
b, a, *extra = signal.butter(order, norm_low, btype='LOW', analog=False)

y = signal.lfilter(b, a, x)

X = fftshift(fft(x))
X_mag = np.abs(X)
f_x = fftshift(fftfreq(len(x)) * fs)

Y = fftshift(fft(y))
Y_mag = np.abs(Y)
f_y = fftshift(fftfreq(len(y)) * fs)

# plot original and filtered square wave
fig1, ax1 = plt.subplots()
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlim(0, 4/f0)
plt.axvline(0, linewidth=1, color='black')
plt.axhline(0, linewidth=1, color='black')
x_major_ticks = np.arange(0, 4/f0, 1/f0)
ax1.set_xticks(x_major_ticks)
ax1.grid(which='major', axis='x', alpha=1)
xml, xsl, xbl = ax1.stem(t, x)
plt.setp(xml, markerfacecolor='none', markeredgewidth='1')
plt.setp(xsl, linewidth='1')
plt.setp(xbl, color='none')
yml, ysl, ybl = ax1.stem(t, y)
plt.setp(yml, markerfacecolor='none', markeredgewidth='1', markeredgecolor='#d65128')
plt.setp(ysl, linewidth='1', color='#d65128')
plt.setp(ybl, color='none')
ax1.set_xlabel('Time (s)')
ax1.set_title('Discrete-Time Square Wave Filtered with Low-Pass IIR')

# plot freq spectrum
fig2, ax2 = plt.subplots()
ax2.plot(f_x, X_mag)
ax2.plot(f_y, Y_mag)
ax2.set_xlim(-fs/2, fs/2)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Spectrum')

# plot filter freq response
w, h = signal.freqz(b, a, worN=8000)
fig3, ax3 = plt.subplots()
freqs = w * fs/(2*np.pi)
ax3.plot(freqs, np.abs(h))
ax3.set_xlim(0, 400)
ax3.set_title("Lowpass Filter Frequency Response")
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Gain (dB)')
ax3.grid()

plt.show()
