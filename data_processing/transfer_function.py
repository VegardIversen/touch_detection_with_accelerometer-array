from cProfile import label
import numpy as np
from csv_to_df import csv_to_df
from scipy import signal
import matplotlib.pyplot as plt
from data_processing.preprocessing import crop_data
def transfer_function():
    SAMPLE_RATE = 150000
    CHANNEL_NAMES_CHIRP = ['channel 1', 'channel 2', 'channel 3', 'chirp']
    CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']
    chirp_df = csv_to_df(file_folder='div_files',
                        file_name='chirp_test_fs_150000_t_max_1s_20000-60000hz_1vpp_1cyc_setup3_v3', channel_names=CHANNEL_NAMES_CHIRP)

    chirp_gen_df = csv_to_df(file_folder='div_files',
                                file_name='chirp_custom_fs_150000_tmax_1_20000-60000_method_linear', channel_names=['chirp'])

    #b, a = signal.butter(5, 1000 / (SAMPLE_RATE / 2), btype='highpass', output='ba')
    #filt_touch = signal.filtfilt(b, a, touch_df['channel 1'])
    chirp = chirp_df['chirp']
    chirp_gen = chirp_gen_df['chirp']
    chirp_df_signals = chirp_df.drop(columns=['channel 2','chirp'], axis=1)
    chirp_df_signals_cropped = crop_data(chirp_df_signals, 0.5, 2)
    #plt.plot(chirp_df_signals_cropped)
    sig_corr = signal.correlate(chirp_df_signals_cropped['channel 1'], chirp, mode='full')
    b, a = signal.butter(2,[2*1000/(SAMPLE_RATE/2),5*1000/(SAMPLE_RATE/2)] , btype='bandstop', output='ba')
    filt_y = signal.filtfilt(b, a, chirp_gen)

    T, yout = signal.impulse((b, a), T=np.linspace(0, 5, SAMPLE_RATE))
    #plt.plot(T, filt_y, label='filtered chirp')
    #plt.plot(T, yout, label='impulse response')
    print(len(sig_corr))
    plt.plot(sig_corr[675000:675400], label='correlation')
    #plt.plot(crop_data(chirp_df_signals, 0.3, 3))
    plt.legend()
    plt.show()


# chirp_cropped = crop_data_threshold(chirp_df.iloc[:stop])
# x = np.correlate(chirp_cropped['channel 1'], chirp_gen_df['channel 1'], 'full')
# ax1 = plt.subplot(211)
# plt.plot(chirp_df['channel 1'])
# plt.subplot(212)
# plt.plot(chirp_gen_df['channel 1'])
# b, a = scipy.signal.butter(2,[2,5]*1000//(SAMPLE_RATE*2))