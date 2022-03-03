import matplotlib.pyplot as plt
import numpy as np
plt.style.use("bmh")

n = 51

windows = {
    "Rectangular": np.ones(n),
    "Hanning": np.hanning(n),
    "Hamming": np.hamming(n),
    "Bartlett": np.bartlett(n),
    "Blackman": np.blackman(n),
    "Kaiser": np.kaiser(n, 14)
}


def plot(position, window_name, xlabel=False):
    position.plot(windows[window_name])
    position.set_title(f"{window_name} window")
    if xlabel:
        position.set_xlabel("$n$")
    position.set_ylabel("$w(t)$")


fig, ax = plt.subplots(3, 2)

plot(position=ax[0, 0], window_name="Hamming")
plot(position=ax[0, 1], window_name="Hanning")
plot(position=ax[1, 0], window_name="Blackman")
plot(position=ax[1, 1], window_name="Bartlett")
plot(position=ax[2, 0], window_name="Rectangular", xlabel=True)
plot(position=ax[2, 1], window_name="Kaiser", xlabel=True)
plt.show()
