from data_viz_files.visualise_data import compare_signals
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
import pandas as pd
from csv_to_df import csv_to_df
from data_processing.find_propagation_speed import find_propagation_speed


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 5


def main():
    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_150000_t_max_2s_1000-40000hz_1vpp_1cyc_setup3_v1')

    find_propagation_speed(chirp_df, sr=SAMPLE_RATE)

    # noise_df = csv_to_df(file_folder='base_data',
    #                      file_name='df_average_noise',
    #                      channel_names=None)

    # df = csv_to_df(file_folder='first_test_touch_passive_setup2',
    #                file_name='touch_test_passive_setup2_place_A1_center_v1')

    # reduced_noise = noise_reduce_signal(df['channel 1'], noise_df['channel 1'])

    # compare_signals(df['channel 1'],
    #                 pd.Series(reduced_noise, copy=False),
    #                 save=True,
    #                 filename='signal_and_reduced_nois_signal.png')


if __name__ == '__main__':
    main()
