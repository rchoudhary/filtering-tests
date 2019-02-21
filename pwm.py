import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from scipy import signal

fpwm = 10000  # pwm freq (Hz)
f0 = 100  # fundamental freq (Hz)
fs = 50000
T = 3  # observation length (s)
Z = T * 100  # zero stretch length (s)
t = np.linspace(0, T, fs * T)
duty = 0.5 + 0.5 * np.sin(2 * np.pi * f0 * t)

# pwm wave with frequency fpwm with a duty cycle that varies according to a cos w/ frequency f0
x = 0.5 + 0.5 * signal.square(2 * np.pi * fpwm * t, duty)

# apply rc filter
RC_filter = ([1], [0.003, 1])
tout, y, tmp = signal.lsim(RC_filter, x, t, interp=False)

# apply windowing function
window = signal.windows.blackmanharris(len(x))
x_windowed = x * window
y_windowed = y * window

# zero padding
x_windowed = np.pad(x_windowed, (0, int(Z * fs)), mode='constant')
y_windowed = np.pad(y_windowed, (0, int(Z * fs)), mode='constant')

X = fftshift(fft(x_windowed))
X_db = 20 * np.log10(np.abs(X) + 1)
f_x = fftshift(fftfreq(len(x_windowed)) * fs)

Y = fftshift(fft(y_windowed))
Y_db = 20 * np.log10(np.abs(Y) + 1)
f_y = fftshift(fftfreq(len(y_windowed)) * fs)

# plot original and filtered pwm wave
fig1, ax1 = plt.subplots()
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlim(1 / f0, 2 / f0)
plt.axvline(0, linewidth=1, color='black')
plt.axhline(0, linewidth=1, color='black')
x_minor_ticks = np.arange(1 / f0, 2 / f0, 10/fpwm)
ax1.set_xticks(x_minor_ticks, minor=True)
ax1.grid(which='minor', axis='x', alpha=0.5)
ax1.plot(t, x)
ax1.plot(t, y)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Value')
ax1.set_title('Square Wave Filtered')

# plot freq spectrum
fig2, ax2 = plt.subplots()
ax2.plot(f_x, X_db)
ax2.plot(f_y, Y_db)
ax2.set_xlim(-25, 1025)
x_minor_ticks = np.arange(0, 1000, 100)
ax2.set_xticks(x_minor_ticks, minor=True)
ax2.grid(which='minor', axis='x', alpha=0.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.set_title('Frequency Spectrum')

# plot RC filter freq response
w, h = signal.freqresp(RC_filter)
fig3, ax3 = plt.subplots()
freqs = w / (2*np.pi)
h_db = 20 * np.log10(np.abs(h))
ax3.set_xscale('log')
ax3.set_title("RC Filter Frequency Response")
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Magnitude (dB)')
ax3.grid()
ax3.plot(freqs, h_db)

plt.show()
