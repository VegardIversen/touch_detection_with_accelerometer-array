import numpy as np
from scipy import signal


def high_pass_filter(sig, cutoff=1000, fs=150000, order=5):
    """High pass filter a signal."""
    b, a = signal.butter(order, cutoff / (fs / 2), btype='highpass')
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig


def low_pass_filter(sig, cutoff=1000, fs=150000, order=5):
    """Low pass filter a signal."""
    b, a = signal.butter(order, cutoff / (fs / 2), btype='lowpass')
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig
