import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from pathlib import Path
from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES
from data_processing import cross_correlation_position as ccp
from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals, plot_vphs, plot_fft, plot_plate_speed_sliders_book, plot_estimated_reflections_with_sliders
from data_processing.preprocessing import crop_data, filter_general, compress_chirp, get_phase_and_vph_of_compressed_signal,cut_out_signal, manual_cut_signal, compress_df_touch
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates
import timeit
if __name__ == '__main__':
    CROSS_CORR_PATH1 = '\\vegard_og_niklas\\setup2_korrelasjon\\'
    CROSS_CORR_PATH2 = '\\first_test_touch_passive_setup2\\'
    #set base new base after new measurements:
    #execution_time = timeit.timeit(ccp.SetBase, number=1)
    #print(f'Execution time: {execution_time}')
    ccp.SetBase(CROSS_CORR_PATH2)
    #ccp.run_test(tolatex=True)
    ccp.run_test(tolatex=True,data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\',filename='results_correlation_old_samples.csv')
    #find position
    #ccp.FindTouchPosition(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\vegard_og_niklas\\setup2_korrelasjon\\A2_v3.csv')

    # List all the files in the test folder that end with V2 or V3
    # test_folder = Path(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\vegard_og_niklas\\setup2_korrelasjon\\')
    # #test_folder = Path('path_to_test_folder')
    # test_files = test_folder.glob('*[v2|v3].csv')

    # # Initialize a data frame to store the results
    # results = pd.DataFrame()

    # # Iterate over all the test files
    # for file in test_files:
    #     # Extract the true label from the file name
    #     true_label = file.name[:2]
    #     # Apply the FindTouchPosition function to the file
    #     predicted_label, direction_check_used = ccp.FindTouchPosition(file)
    
    #     print(f'direction_check_used: {direction_check_used}')
    #     # Add the results to the data frame
    #     result_df = pd.DataFrame({'file':file.name,'true_label': true_label, 'predicted_label': predicted_label, 'direction_check': direction_check_used}, index=[0])
    #     result_df['direction_check'] = result_df['direction_check'].astype(bool)
    #     # Concatenate the result DataFrame to the main DataFrame
    #     results = pd.concat([results, result_df])
    #     # Save the results to a CSV file
    # results.to_csv('results.csv', index=False)

