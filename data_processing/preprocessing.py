from scipy import signal
import pandas as pd

"""FILTERING"""


def hp_or_lp_filter(sig, filtertype, cutoff=1000, fs=150000, order=8):
    """filtertype: 'highpass' or 'lowpass'"""

    b, a = signal.butter(order, cutoff / (fs / 2), btype=filtertype)

    sig_filtered = sig
    if isinstance(sig, pd.DataFrame):
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.filtfilt(b,
                                                    a,
                                                    sig[channel].values)
    else:
        sig_filtered = signal.filtfilt(b,
                                       a,
                                       sig)



    return sig_filtered


def filter_signal(sig, freqs, sample_rate=150000):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        q = freq ** (1 / 3) # We want smaller q-factors for higher frequencies
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * sample_rate), q)
        sig_filtered = sig
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.filtfilt(b_notch,
                                                    a_notch,
                                                    sig[channel].values)
    return sig_filtered


"""CROPPING"""


def crop_data(df, time_start=0, time_end=5, sample_rate=150000):
    """Crop data to the range given by the
    global variables CROP_START and CROP_END.
    """
    data_cropped = df.truncate(before=(time_start * sample_rate),
                               after=(time_end * sample_rate))
    return data_cropped


def crop_data_threshold(data, threshold=0.0006):
    if isinstance(data, pd.DataFrame):
        data_cropped = data.loc[(data > threshold).any(axis=1)]
    else:
        data_cropped = data.loc[data > threshold]
    return data_cropped
