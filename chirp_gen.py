import scipy.signal as signal
import numpy as np


def generate_chirp_and_save_to_file(Fs=150000,
                                    freq_start=20000,
                                    freq_stop=40000,
                                    t_max=0.125,
                                    scale='linear'):
    print(f"Fs: {Fs}, freq_start: {freq_start}, freq_stop: {freq_stop}, t_max: {t_max}")
    print(f"Time-bandwidth product: {t_max * Fs}")

    time_axis = np.linspace(0, t_max, int(t_max * Fs))
    generated_chirp = signal.chirp(t=time_axis,
                                   f0=freq_start,
                                   t1=t_max,
                                   f1=freq_stop,
                                   method=scale)
    np.savetxt(f"chirp_{Fs}Hz_{t_max}s_{freq_start}Hz-{freq_stop}Hz_{scale}_method.csv",
               generated_chirp,
               delimiter=",")
