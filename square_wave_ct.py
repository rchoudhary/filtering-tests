import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from scipy import signal

f0 = 100  # square wave freq (Hz)
fs = 5000  # sampling frequency (Hz)
T = 30/f0 + 0.002319  # observation length (s)
Z = T * 50  # zero stretch length (s)
t = np.linspace(0, T, int(fs * T))
x = 0.5 + 0.5 * signal.square(2 * np.pi * f0 * t)  # square wave with frequency f0

# apply RC filter
RC_filter = ([1], [0.008, 1])
tout, y, tmp = signal.lsim(RC_filter, x, t, interp=False)

# apply windowing function
window = signal.windows.blackmanharris(len(x))
x_windowed = x * window
y_windowed = y * window

# zero padding
x_windowed = np.pad(x_windowed, (0, int(Z * fs)), mode='constant')
y_windowed = np.pad(y_windowed, (0, int(Z * fs)), mode='constant')

X = np.abs(fftshift(fft(x_windowed)))
X_db = 20 * np.log10(np.abs(X) + 1)
f_x = fftshift(fftfreq(len(x_windowed)) * fs)

Y = fftshift(fft(y_windowed))
Y_db = 20 * np.log10(np.abs(Y) + 1)
f_y = fftshift(fftfreq(len(y_windowed)) * fs)

# plot square wave
fig1, ax1 = plt.subplots()
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlim(0, 10/f0)
plt.axvline(0, linewidth=1, color='black')
plt.axhline(0, linewidth=1, color='black')
x_major_ticks = np.arange(0, 10/f0, 1/f0)
ax1.set_xticks(x_major_ticks)
ax1.grid(which='major', axis='x', alpha=1)
ax1.plot(t, x)
ax1.plot(t, y)
ax1.set_xlabel('Time (s)')
ax1.set_title('Square Wave')

# plot freq spectrum
fig2, ax2 = plt.subplots()
plt.axhline(0, linewidth=0.8, color='black')
ax2.plot(f_x, X_db)
ax2.plot(f_y, Y_db)
x_minor_ticks = np.arange(-fs / 2, fs / 2, f0)
ax2.set_xticks(x_minor_ticks, minor=True)
ax2.set_xlim(-30, 600)
ax2.set_ylim(0, 50)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.set_title('FFT')

# plot RC filter frequency response
w, h = signal.freqresp(RC_filter)
fig3, ax3 = plt.subplots()
freqs = w / (2*np.pi)
h_db = 20 * np.log10(np.abs(h))
ax3.set_xscale('log')
ax3.set_title("RC Filter Frequency Response")
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Gain (dB)')
ax3.grid()
ax3.plot(freqs, h_db)

plt.show()
