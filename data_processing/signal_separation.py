import numpy as np
from csv_to_df import csv_to_df
from scipy import signal
import matplotlib.pyplot as plt
from data_processing.preprocessing import crop_data, filter_general

def signal_sep():
    SAMPLE_RATE = 150000
    CHANNEL_NAMES_CHIRP = ['channel 1', 'channel 2', 'channel 3', 'chirp']
    CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']

    chirp_df = csv_to_df(file_folder='div_files',
                        file_name='chirp_test_fs_150000_t_max_1s_20000-60000hz_1vpp_1cyc_setup3_v3', channel_names=CHANNEL_NAMES_CHIRP)

    chirp_gen_df = csv_to_df(file_folder='div_files',
                                file_name='chirp_custom_fs_150000_tmax_1_20000-60000_method_linear', channel_names=['chirp'])

    # b, a = signal.butter(5, 1000 / (SAMPLE_RATE / 2), btype='highpass', output='ba')
    # filt_touch = signal.filtfilt(b, a, touch_df['channel 1'])
    chirp = chirp_df['chirp']
    close_to_chirp = chirp_df['channel 2']
    close_to_chirp_filt = filter_general(close_to_chirp, 'lowpass', cutoff_lowpass=20000)
    return chirp, chirp_df