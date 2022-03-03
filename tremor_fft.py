import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
plt.style.use("bmh")


data = "small"


mat = scipy.io.loadmat(f'./data/{data}.mat')
y = mat["f0"]
fs = int(mat["samplerate"][0][:-3])
title = mat["dataset"][0]
N = len(y)

n = np.arange(0.0, N)
T = 1.0 / fs
t = n*T
delta_f = fs/N

yf = np.abs(fft(y))
with np.errstate(divide='ignore', invalid='ignore'):
    ydb = 20 * np.log10(yf)
nf = np.arange(0, N)
ff = nf * fs/N

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y)
ax[0].set_xlabel("Time/s")
ax[0].set_ylabel("$x(t)$")
ax[0].set_title(title)

ax[1].plot(ff, 2.0/N * yf[:N])
ax[1].set_xlabel("Frequency/Hz")

ax[1].set_ylabel(f"DFT $X(f)$")
ax[2].set_ylabel(f"DFT $X(f)/ dB$")

ax[2].plot(ff, 2.0/N * ydb[:N])
ax[2].set_xlabel("Frequency/Hz")

plt.legend()
plt.show()
