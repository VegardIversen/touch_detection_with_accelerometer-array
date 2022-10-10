from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from csv_to_df import csv_to_df
from data_processing.find_propagation_speed import find_propagation_speed
from data_processing.detect_echoes import find_indices_of_peaks
from data_processing.preprocessing import crop_data
from data_processing.transfer_function import transfer_function


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 1.46
TIME_END = 2

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']
def main():
    transfer_function()
    #chirp_df = csv_to_df(file_folder='div_files',
    #                      file_name='chirp_test_fs_96000_t_max_2s_2000-20000hz_1vpp_1cyc_setup3_v2', channel_names=CHIRP_CHANNEL_NAMES)

    #signal_df = crop_data(signal_df, time_start=TIME_START, time_end=TIME_END)

    #compare_signals(signal_df['channel 1'], signal_df['channel 1'], sample_rate=SAMPLE_RATE, time_start=TIME_START, time_end=TIME_END)
    #find_propagation_speed(chirp_df, sr=SAMPLE_RATE)

if __name__ == '__main__':
    main()
