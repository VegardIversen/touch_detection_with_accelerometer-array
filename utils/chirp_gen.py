"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import scipy.signal as signal
import numpy as np


def generate_chirp(sample_rate=150000,
                   frequency_start=20000,
                   frequency_stop=40000,
                   time_end=0.125,
                   chirp_method='linear',
                   save_to_file=True):
    print(f"Fs: {sample_rate}, freq_start: {frequency_start}, freq_stop: {frequency_stop}, t_max: {time_end} s")

    time_axis = np.linspace(0, time_end, int(time_end * sample_rate))
    generated_chirp = signal.chirp(t=time_axis,
                                   f0=frequency_start,
                                   t1=time_end,
                                   f1=frequency_stop,
                                   method=chirp_method)
    if save_to_file:
        np.savetxt(f"chirp_{sample_rate}Hz_{time_end}s_{frequency_start}Hz-{frequency_stop}Hz_{chirp_method}.csv",
                   generated_chirp,
                   delimiter=",")

    return generated_chirp
