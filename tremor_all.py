import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
plt.style.use("bmh")

data = ["small", "medium", "large"]
fig, ax = plt.subplots(3, 3)


def plot(ax, data, i):
    mat = scipy.io.loadmat(f'./data/{data}.mat')
    y = mat["f0"][:10000]
    y = np.concatenate(y, axis=0)
    fs = int(mat["samplerate"][0][:-3])
    title = mat["dataset"][0]
    N = len(y)
    cutter = N//35

    n = np.arange(0.0, N)
    T = 1.0 / fs
    t = n*T

    yf = np.abs(fft(y))
    with np.errstate(divide='ignore', invalid='ignore'):
        ydb = 20 * np.log10(yf)
    nf = np.arange(0, N)[:cutter]
    ff = nf * fs/N

    ax[0][i].plot(t, y)
    ax[0][i].set_xlabel("Time/s")
    ax[0][i].set_ylabel("$x(t)$")
    ax[0][i].set_title(title)
    ax[1][i].plot(ff, 2.0/N * yf[:cutter])
    ax[1][i].set_xlabel("Frequency/Hz")
    ax[1][i].set_ylabel(f"DFT $X(f)$")
    ax[2][i].set_ylabel(f"DFT $X(f) [dB]$")
    ax[2][i].plot(ff, 2.0/N * ydb[:cutter])
    ax[2][i].set_xlabel("Frequency/Hz")


for i in range(3):
    plot(ax, data[i], i)
plt.show()
