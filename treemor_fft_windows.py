import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
plt.style.use("bmh")


data = "small"


mat = scipy.io.loadmat(f'./data/{data}.mat')
y = np.concatenate(mat["f0"][0:6000], axis=0)
fs = int(mat["samplerate"][0][:-3])
title = mat["dataset"][0]
N = len(y)
windows = {
    "Rectangular": np.ones(N),
    "Hanning": np.hanning(N),
    # "Hamming": np.hamming(N),
    # "Bartlett": np.bartlett(N),
    "Blackman": np.blackman(N),
    # "Kaiser": np.kaiser(N, 14)
}
y_win_dict = {}
y_win_dict_db = {}

n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N

for key in windows.keys():
    y_win = np.multiply(y, windows[key])
    yf = np.abs(fft(y_win))
    with np.errstate(divide='ignore', invalid='ignore'):
        ydb = 20 * np.log10(yf)
    y_win_dict_db[key] = ydb[:N//30]
    y_win_dict[key] = yf[:N//30]

ff = fftfreq(N, T)[:N//30]

fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1.5, 1]})
ax[0].plot(t, y)
ax[0].set_xlabel("Time/s")
ax[0].set_ylabel("$x(t)$")
ax[0].set_title(title)

for key in y_win_dict.keys():
    ax[1].plot(ff, y_win_dict[key], label=key)

ax[1].set_xlabel("Frequency/Hz")
ax[1].set_ylabel("DFT $X(f)$")
ax[2].set_ylabel("DFT $X(f)$ [dB]")

for key in y_win_dict.keys():
    ax[2].plot(ff, y_win_dict_db[key], label=key)
ax[2].set_xlabel("Frequency/Hz")
ax[1].legend()
ax[2].legend()

plt.legend()
plt.show()
