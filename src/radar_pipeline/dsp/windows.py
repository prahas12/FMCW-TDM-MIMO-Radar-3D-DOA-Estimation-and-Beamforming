import numpy as np

def get_window(name, size):
    if name == 'hann':
        return np.hanning(size)
    elif name == 'hamming':
        return np.hamming(size)
    else:
        return np.ones(size)
