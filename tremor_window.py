import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
plt.style.use("bmh")


data = "small"


mat = scipy.io.loadmat(f'./data/{data}.mat')
y = np.concatenate( mat["f0"][0:6000], axis=0)
fs = int(mat["samplerate"][0][:-3])
title = mat["dataset"][0]
N = len(y)
window = np.hanning(N)
y_win = np.multiply(y, window)
n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N

yf = np.abs(fft(y))
yf_window = np.abs(fft(y_win))
# print(yf)
with np.errstate(divide='ignore', invalid='ignore'):
    ydb = 20 * np.log10(yf)
    ydb_win = 20 * np.log10(yf_window)
ff = fftfreq(N, T)[:N//30]

fig, ax = plt.subplots(3, 2)
ax[0,0].plot(t, y)
ax[0,0].set_xlabel("Time/s")
ax[0,0].set_ylabel("$x(t)$")
ax[0,0].set_title(title)

ax[1,0].plot(ff, yf[:N//30])
ax[1,0].set_xlabel("Frequency/Hz")

ax[1,0].set_ylabel(f"DFT $X(f)$")
ax[2,0].set_ylabel(f"DFT $X(f) [dB]$")

ax[2,0].plot(ff,  ydb[:N//30])
ax[2,0].set_xlabel("Frequency/Hz")

ax[0,1].plot(t, y_win)
ax[0,1].set_xlabel("Time/s")
ax[0,1].set_ylabel("$x(t)$")
ax[0,1].set_title(title)
ax[1,1].plot(ff, yf_window[:N//30])
ax[1,1].set_xlabel("Frequency/Hz")

ax[1,1].set_ylabel(f"DFT $X(f)$")
ax[2,1].set_ylabel(f"DFT $X(f) [dB]$")

ax[2,1].plot(ff,  ydb_win[:N//30])
ax[2,1].set_xlabel("Frequency/Hz")

plt.legend()
plt.show()
