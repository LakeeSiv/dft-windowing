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
print(y)
n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N

yf = np.abs(fft(y))
# print(yf)
with np.errstate(divide='ignore', invalid='ignore'):
    ydb = 20 * np.log10(yf)
ff = fftfreq(N, T)[:N//30]

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y)
ax[0].set_xlabel("Time/s")
ax[0].set_ylabel("$x(t)$")
ax[0].set_title(title)

ax[1].plot(ff, yf[:N//30])
ax[1].set_xlabel("Frequency/Hz")

ax[1].set_ylabel(f"DFT $X(f)$")
ax[2].set_ylabel(f"DFT $X(f) [dB]$")

ax[2].plot(ff,  ydb[:N//30])
ax[2].set_xlabel("Frequency/Hz")

plt.legend()
plt.show()
