import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from data_processing.direction_of_arrival import degree_calc
from data_processing.preprocessing import cut_out_signal, cut_out_signal_df
#from direction_of_arrival import degree_calc
from statistics import mode
import matplotlib.pyplot as plt
import scipy


SAMPLE_RATE = 150000     # Hz

CROP_MODE = "Auto"      # Auto or Manual
CROP_BEFORE = 80000     # samples
CROP_AFTER = 120000     # samples

DATA_DELIMITER = ","
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'wave_gen']
DIRECTION_CHECK = {
                    'A1': range(190,260), 'A2': range(250,290), 'A3': range(280,350),
                    'B1': range(180,250), 'B2': range(250,290), 'B3': range(290,350),
                    'C1': range(100,190), 'C2': range(250,290), 'C3': range(350,60)}
GRID_NAMES = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
RESULT_DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\test_results\\'
OUTPUT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\grid_data.pkl'
Path(OUTPUT_FOLDER).parent.mkdir(parents=True, exist_ok=True)
print(OUTPUT_FOLDER)


def extract_touch_signals(df, seconds_before=0, threshold=0.00065):
    """
    Extracts the touch signal from a dataframe containing multiple channels.

    Parameters:
    - df (pandas DataFrame): the dataframe containing the channels
    - seconds_before (float): the number of seconds of data before the touch to include in the extracted dataframe (default: 0)
    - threshold (float): the minimum value to consider a signal as a touch (default: 0.1)

    Returns:
    - touch_df (pandas DataFrame): a dataframe containing only the rows corresponding to the touch signal
    - touch_start_time (float): the start time of the touch signal (in seconds)
    - touch_end_time (float): the end time of the touch signal (in seconds)
    """
    # Find the start and end times of the touch
    # You can do this by looking for the first and last values above the threshold in each channel of the dataframe
    touch_starts = df[df > threshold].apply(lambda x: x.idxmax(), axis=0)
    touch_ends = df[df > threshold].apply(lambda x: x.idxmin(), axis=0)

    # Find the earliest start time and the latest end time among all channels
    touch_start_index = touch_starts.min()
    touch_end_index = touch_ends.max()

    # Calculate the start and end times of the touch in seconds
    time_stamps = df.index.values
    touch_start_time = df.index.get_loc(touch_start_index, method='nearest') / 1e6
    touch_end_time = df.index.get_loc(touch_end_index, method='nearest') / 1e6

    # Check if either of the start or end times is a NaN value
    if np.isnan(touch_start_time) or np.isnan(touch_end_time):
        # If either of the start or end times is a NaN value, return an empty dataframe
        touch_df = pd.DataFrame()
    else:
        # Otherwise, extract the touch signal from the dataframe using the start and end indices
        start_index = df.index.get_loc(touch_start_index, method='nearest')
        end_index = df.index.get_loc(touch_end_index, method='nearest')
        touch_df = df.loc[start_index:end_index, :]

    return touch_df, touch_start_time, touch_end_time

def pick_relevant_signal(signal, sample_rate, threshold, min_duration):
  """Picks the relevant part of a signal based on a threshold and minimum duration.

  Args:
    signal: 1D array of float or int values.
    sample_rate: float or int value representing the number of samples per second.
    threshold: float or int value. Any value in the signal below this threshold will be ignored.
    min_duration: float or int value representing the minimum duration of the relevant signal, in seconds.

  Returns:
    relevant_signal: 1D array of float or int values, containing the relevant signal.
  """
  relevant_signal = []
  in_relevant_section = False
  for i, value in enumerate(signal):
    if value >= threshold:
      if not in_relevant_section:
        # Start of a new relevant section
        start_index = i
        in_relevant_section = True
    else:
      if in_relevant_section:
        # End of a relevant section
        end_index = i
        duration = (end_index - start_index) / sample_rate
        if duration >= min_duration:
          # Relevant section is long enough, append it to the result
          relevant_signal.extend(signal[start_index:end_index])
        in_relevant_section = False
  # Check if the last section was relevant
  if in_relevant_section:
    end_index = len(signal)
    duration = (end_index - start_index) / sample_rate
    if duration >= min_duration:
      relevant_signal.extend(signal[start_index:end_index])
  return relevant_signal

def CropData(data):
    #get the x value where the threshold is crossed
    #start_crop = data['channel 1'].gt(0.00065).idxmax()
    #data_cropped = data.loc[(data>0.00065).any(axis=1)]
    #data_cropped,_,_ = extract_touch_signals(data, 0.00065)
    data_cropped = cut_out_signal_df(data, 150000, 0.00065)
    #print('done')
    #print(type(data_cropped))
    return data_cropped


def CropRegion(data):
    mask = data['channel 1'].gt(0.1)
    rgns = mask.diff().fillna(True).cumsum()
    gb = data[mask].groupby(rgns)
    gb.plot()
    plt.show()


def SetBase(path='\\first_test_touch_passive_setup2\\', crop=True, detrend=True, cell_names=GRID_NAMES, output_folder=OUTPUT_FOLDER):
    file_path = Path(DATA_FOLDER + path).rglob('*_v1.csv')
    base_dict = {}
    for idx, file in enumerate(sorted(file_path)):
        data = pd.read_csv(file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
        print(data.keys())
        if len(data.keys()) > 3:
            print('removing wave_gen')
            data = data.drop(columns=['wave_gen'])
        if detrend:
            print('detrending')
            data = data.apply(scipy.signal.detrend)
        
        if crop:
            print('cropping')
            data = CropData(data)
        has_nan = data.isnull().any().any()
        if has_nan:
            data.dropna(inplace=True)
        base_dict[cell_names[idx]] = data
    print(output_folder)

    with open(output_folder, 'wb') as f:
        pickle.dump(base_dict, f)


def LoadData(file_loc=OUTPUT_FOLDER):
    with open(file_loc, 'rb') as f:
        loaded_dict = pd.read_pickle(f)
    return loaded_dict


def FindTouchPosition(file, crop=True, direction_check=True, channel='channel 1'):
    test_data = pd.read_csv(file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
    double_checked = False
    if len(test_data.keys()) > 3:
            test_data = test_data.drop(columns=['wave_gen'])
    #CropRegion(test_data)
    highest_score = 0
    grid_cell = ' '
    degrees = 'not used'
    base_dict = LoadData(OUTPUT_FOLDER)
    if crop:
        test_data = CropData(test_data)
    has_nan = test_data.isnull().any().any()
    if has_nan:
        test_data.dropna(inplace=True)
    for cell in GRID_NAMES:
        cc = scipy.correlate(base_dict[cell][channel], test_data[channel], mode='valid')

        max_val = max(cc)
        if np.isnan(max_val):
            print('nan found')
            cc = scipy.correlate(base_dict[cell][channel], test_data[channel], mode='same')
            max_val = max(cc)
        #print(f'max val for {cell} is {max_val}')
        if max_val > highest_score:
            highest_score = max_val
            grid_cell = cell
            #print(grid_cell)
    if direction_check:
        double_checked = False
        degree = degree_calc(test_data)
        degrees = degree
        print(f'degree of signal: {degree}')
        print(f'range of cell: {grid_cell} is {DIRECTION_CHECK[grid_cell]}')
        if int(degree) not in DIRECTION_CHECK[grid_cell]:
            double_checked = True
            print('Probably error in estimation')
            print('Checking different channels')
            grid_cells = [grid_cell]
            start_channel = channel
            for ch in CHANNEL_NAMES[1:-1]:
                ch_res,_,_ = FindTouchPosition(file, direction_check=False, channel=ch)
                grid_cells.append(ch_res)
            
            try:
                print(mode(grid_cells))
                grid_cell = mode(grid_cells)
            except:
                print('uncertain result, this is the guessed cells. Returns the first value')
                print(grid_cells)
                grid_cell = grid_cell

    return grid_cell, double_checked, degrees

def run_test(
            data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\vegard_og_niklas\\setup2_korrelasjon\\',
            pick_files='*[v2|v3].csv',
            output_folder=RESULT_DATA_FOLDER,
            filename='results_correlation.csv',
            tolatex=False,):
    
    test_folder = Path(f'{Path.home()}'+data_folder)
    #test_folder = Path('path_to_test_folder')
    test_files = test_folder.glob(pick_files)

    # Initialize a data frame to store the results
    results = pd.DataFrame()

    # Iterate over all the test files
    for file in test_files:
        # Extract the true label from the file name
        true_label = file.name[:2]
        # Apply the FindTouchPosition function to the file
        predicted_label, direction_check_used, degrees = FindTouchPosition(file)
        #check if degrees is float
        if isinstance(degrees, float):
            degrees = round(degrees, 2)
            
        print(f'direction_check_used: {direction_check_used}')
        # Add the results to the data frame
        if true_label == predicted_label:
            res = 'correct prediction'
        else:
            res = 'wrong prediction'
        
        
        result_df = pd.DataFrame({'file':file.name,'true_label': true_label, 'predicted_label': predicted_label,'Result': res, 'direction_check': direction_check_used, 'degrees calculated': degrees}, index=[0])
        result_df['direction_check'] = result_df['direction_check'].astype(bool)
        # Concatenate the result DataFrame to the main DataFrame
        results = pd.concat([results, result_df])
        # Save the results to a CSV file
    if tolatex:
        print(results.to_latex(index=False))
    results.to_csv(filename, index=False)

if __name__ == '__main__':

    #print(pd.__version__)
    #SetBase()
    #dict = LoadData()
    #print(type(dict))
    print(FindTouchPosition(DATA_FOLDER + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A2_center_v2.csv'))
    #CropRegion()
