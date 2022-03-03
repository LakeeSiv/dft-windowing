import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import numpy as np


n = 51

windows = {
    "Rectangular": np.ones(n),
    "Hanning": np.hanning(n),
    "Hamming": np.hamming(n),
    "Bartlett": np.bartlett(n),
    "Blackman": np.blackman(n),
    "Kaiser": np.kaiser(n, 14)
}


def calc(window_name):
    window = windows[window_name]
    A = fft(window, 2048) / 25.5
    mag = np.abs(fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    with np.errstate(divide='ignore', invalid='ignore'):
        response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)

    return freq, response


plt.style.use("bmh")
fig, ax = plt.subplots(3, 2)


def plot(position, window_name, xlabel=False):
    freq, response = calc(window_name)
    position.plot(freq, response)
    position.set_title(f"Frequency response of the {window_name} window")
    if xlabel:
        position.set_xlabel("Normalized frequency [cycles per sample]")
    position.set_ylabel("Magnitude [dB]")


plot(position=ax[0, 0], window_name="Hamming")
plot(position=ax[0, 1], window_name="Hanning")
plot(position=ax[1, 0], window_name="Blackman")
plot(position=ax[1, 1], window_name="Bartlett")
plot(position=ax[2, 0], window_name="Rectangular", xlabel=True)
plot(position=ax[2, 1], window_name="Kaiser", xlabel=True)
plt.show()
