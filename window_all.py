import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fft import fft
plt.style.use("bmh")

N = 7000  # number of samples
fs = 500  # sampling frequency
max_f = 10  # max frequency plotted

n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N


y = (125*np.sin(2*3.41*np.pi*t) + 75*np.sin(7.74 *
                                            2.0*np.pi*t) + 43*np.sin(7.17 * 2.0*np.pi*t))*np.exp(-t/15)
windows = {
    "Rectangular": np.ones(N),
    "Hanning": np.hanning(N),
    "Hamming": np.hamming(N),
    "Bartlett": np.bartlett(N),
    "Blackman": np.blackman(N),
    "Kaiser": np.kaiser(N, 14)
}
y_win_dict = {}
for key in windows.keys():
    y_win = np.multiply(y, windows[key])
    yf = np.abs(fft(y_win))
    with np.errstate(divide='ignore', invalid='ignore'):
        ydb = 20 * np.log10(yf)
    y_win_dict[key] = ydb[:int(max_f / delta_f)]

nf = np.arange(0, int(max_f / delta_f))
ff = nf * fs/N

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
ax[0].plot(t, y)
ax[0].set_xlabel("Time/s")
ax[0].set_ylabel("$x(t)$")
ax[0].set_title(
    "$x(t) = [125 \sin{(2\pi * 3.41 * t) + 75 \sin{(2\pi * 7.74 * t)}} + 43 \sin{(2\pi * 7.17 * t)}]e^{-t/15}$")
ax[1].set_xlabel("Frequency/Hz")
ax[1].set_ylabel("DFT $X(f)$ [dB]")

for key in y_win_dict.keys():
    ax[1].plot(ff, y_win_dict[key], label=key)
ax[1].legend()
plt.legend()
plt.legend()
plt.show()
