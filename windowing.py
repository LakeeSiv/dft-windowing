import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fft import fft
plt.style.use("bmh")

N = 8000  # number of samples
fs = 1000  # sampling frequency
max_f = 10  # max frequency plotted
decibels = True
stem_plot = True
# window = np.blackman(N)  # windowing function
window = None

n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N


y = np.sin(5.15 * 2.0*np.pi*t) + 0.25*np.sin(5.5 * 2.0*np.pi*t)
# y = 0.25*np.sin(5.5 * 2.0*np.pi*t)
if window is not None:
    y = np.multiply(y, window)
yf = np.abs(fft(y))

with np.errstate(divide='ignore', invalid='ignore'):
    ydb = 20 * np.log10(yf)
nf = np.arange(0, int(max_f / delta_f))
ff = nf * fs/N

if decibels:
    fig, ax = plt.subplots(3, 1)
else:
    fig, ax = plt.subplots(2, 1)
ax[0].plot(t, y)
ax[0].set_xlabel("Time/s")
ax[0].set_ylabel("$x(t)$")
ax[0].set_title(
    "$x(t) = \sin{(2\pi * 5.15 * t) + 0.25\sin{(2\pi * 5.5 * t)}}$")
if stem_plot:
    ax[1].stem(ff, 2.0/N * yf[:int(max_f / delta_f)],
               use_line_collection=True)
else:
    ax[1].plot(ff, 2.0/N * yf[:int(max_f / delta_f)])
ax[1].set_xlabel("Frequency/Hz")
ax[1].set_ylabel(f"DFT $X(f)$")
if decibels:
    ax[2].plot(ff, ydb[:int(max_f / delta_f)])


ax[1].axvline(5.15, color="black", linestyle="--", label="5.15 Hz")
ax[1].axvline(5.5, color="gray", linestyle="--", label="5.5 Hz")
plt.legend()
plt.show()
