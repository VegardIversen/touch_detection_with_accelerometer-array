from scipy import signal
import pandas as pd
import numpy as np

"""FILTERING"""


def filter_general(sig, filtertype, cutoff_lowpass=20000, cutoff_highpass=40000, fs=150000, order=8):
    """filtertype: 'highpass', 'lowpass' or 'bandpass"""
    if filtertype == 'highpass':
        b, a = signal.butter(order, cutoff_lowpass / (0.5 * fs), 'highpass')
    elif filtertype == 'lowpass':
        b, a = signal.butter(order, cutoff_highpass / (0.5 * fs), 'lowpass')
    elif filtertype == 'bandpass':
        b, a = signal.butter(order, [cutoff_lowpass / (0.5 * fs),
                             cutoff_highpass / (0.5 * fs)],
                             'bandpass')
    else:
        raise ValueError('Filtertype not recognized')

    sig_filtered = sig.copy()
    if isinstance(sig, pd.DataFrame):
        for channel in sig_filtered:
            sig_filtered[channel] = signal.filtfilt(b,
                                                    a,
                                                    sig[channel].values)
    else:
        sig_filtered = signal.filtfilt(b,
                                       a,
                                       sig)

    return sig_filtered


def filter_notches(sig, freqs, sample_rate=150000):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        q = freq ** (1 / 3) # We want smaller q-factors for higher frequencies
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * sample_rate), q)
        sig_filtered = sig.copy()
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.filtfilt(b_notch, a_notch,
                                                    sig[channel].values)
    return sig_filtered


"""CROPPING"""


def crop_data(sig, time_start=None, time_end=None, threshold=0, sample_rate=150000):
    """Crop either DataFrame input, pandas series or a numpy array input.
    NOTE:   Vegard will fix this to work properly,
            but could use some logic from here.
    """
    # Some logic for assuming cropping type and length
    if (time_start or time_start == 0) and not time_end:
        time_end = len(sig) / sample_rate
    elif time_end and not (time_start or time_start == 0):
        time_start = 0

    if (time_start or time_start ==0) and time_end:
        if isinstance(sig, np.ndarray):
            data_cropped = sig[int(time_start * sample_rate):int(time_end * sample_rate)]
        else:
            # data_cropped = input.loc[time_start * sample_rate:time_end * sample_rate]
            data_cropped = sig.truncate(before=(time_start * sample_rate),
                                        after=(time_end * sample_rate))
    elif not (time_start or time_start == 0) and not time_end:
        if isinstance(sig, pd.DataFrame):
            data_cropped = sig.loc[(sig > threshold).any(axis=1)]
            data_cropped = sig.loc[(sig.iloc[::-1] > threshold).any(axis=1)]
        else:
            data_cropped = sig.loc[sig > threshold]

    return data_cropped


def crop_data_threshold(data, threshold=0.0006):
    if isinstance(data, pd.DataFrame):
        data_cropped = data.loc[(data > threshold).any(axis=1)]
    else:
        data_cropped = data.loc[data > threshold]
    return data_cropped
