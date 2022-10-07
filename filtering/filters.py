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


def filter_signal(sig, freqs, sample_rate=150000):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        # We want smaller q-factors for higher frequencies
        q = freq ** (1 / 3)
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * sample_rate), q)
        sig_filtered = sig

        for channel in sig_filtered:
            sig_filtered[channel] = signal.filtfilt(b_notch,
                                                    a_notch,
                                                    sig[channel].values)

    return sig_filtered