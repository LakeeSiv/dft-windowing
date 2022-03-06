import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fft import fft
plt.style.use("bmh")

N = 800000  # number of samples
fs = 1000  # sampling frequency
max_f = 10  # max frequency plotted
decibels = True
stem_plot = True
window = "Hamming"  # windowing function

n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N


y = np.sin(np.pi * 2.0*np.pi*t) + 0.25*np.sin(3.25 * 2.0*np.pi*t)
windows = {
    "Rectangular": np.ones(N),
    "Hanning": np.hanning(N),
    "Hamming": np.hamming(N),
    "Bartlett": np.bartlett(N),
    "Blackman": np.blackman(N),
    "Kaiser": np.kaiser(N, 14)
}
y_win = np.multiply(y, windows[window])
yf = np.abs(fft(y))
yf_win = np.abs(fft(y_win))

with np.errstate(divide='ignore', invalid='ignore'):
    ydb = 20 * np.log10(yf)
    ydb_win = 20 * np.log10(yf_win)
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
    ax[2].plot(ff, ydb[:int(max_f / delta_f)], label="Un-Windowed")
    ax[2].plot(ff, ydb_win[:int(max_f / delta_f)], label=f"{window} Windowed")
    ax[2].set_xlabel("Frequency/Hz")
    ax[2].set_ylabel(f"DFT $X(f)$ [dB]")

ax[1].axvline(5.15, color="black", linestyle="--", label="5.15 Hz")
ax[1].axvline(5.5, color="gray", linestyle="--", label="5.5 Hz")
plt.legend()
plt.show()
