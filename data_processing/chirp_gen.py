from os import chdir
import scipy.signal as ss
import pandas as pd
import numpy as np

Fs = 150000
freq_start = 500
freq_stop = 40000
t_max = 2
print(t_max*Fs)
scale = 'linear'
t = np.linspace(0,t_max, int(t_max*Fs))
y = ss.chirp(t, freq_start, t_max, freq_stop, method=scale)
np.savetxt(f"chirp_custom_fs_{Fs}_tmax_{t_max}_{freq_start}-{freq_stop}_method_{scale}.csv", y, delimiter=",")