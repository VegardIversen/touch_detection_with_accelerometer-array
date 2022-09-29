import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from data_processing.direction_of_arrival import degree_calc
from statistics import mode


SAMPLE_RATE = 150000     # Hz

CROP_MODE = "Auto"      # Auto or Manual
CROP_BEFORE = 80000     # samples
CROP_AFTER = 120000     # samples

DATA_DELIMITER = ","
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']
DIRECTION_CHECK = {
                    'A1': range(190,260), 'A2': range(250,290), 'A3': range(280,350),
                    'B1': range(180,250), 'B2': range(250,290), 'B3': range(290,350),
                    'C1': range(100,190), 'C2': range(250,290), 'C3': range(350,60)}
GRID_NAMES = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
OUTPUT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\grid_data.pkl'
Path(OUTPUT_FOLDER).parent.mkdir(parents=True, exist_ok=True)
print(OUTPUT_FOLDER)


def CropData(data):
    data_cropped = data.loc[(data>0.0006).any(axis=1)]
    return data_cropped


def SetBase(path='\\first_test_touch_passive_setup2\\', crop=True, cell_names=GRID_NAMES, output_folder=OUTPUT_FOLDER):
    file_path = Path(DATA_FOLDER + path).rglob('*_v1.csv')
    base_dict = {}
    for idx, file in enumerate(sorted(file_path)):
        
        data = pd.read_csv(file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
        if crop:
            data = CropData(data)
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
    highest_score = 0
    grid_cell = ' '
    base_dict = LoadData(OUTPUT_FOLDER)
    if crop:
        test_data = CropData(test_data)
    for cell in GRID_NAMES:
        cc = np.correlate(base_dict[cell][channel], test_data[channel], mode='valid')
        max_val = max(cc)
        if max_val > highest_score:
            highest_score = max_val
            grid_cell = cell
    if direction_check:
        degree = degree_calc(test_data)
        if int(degree) in DIRECTION_CHECK[grid_cell]:
            print('Probably error in estimation')
            print('Checking different channels')
            grid_cells = [grid_cell]
            start_channel = channel
            for ch in CHANNEL_NAMES[1:]:
                grid_cells.append(FindTouchPosition(file, direction_check=False, channel=ch))
            print(grid_cells)
            print(mode(grid_cells))
            return mode(grid_cells)


    return grid_cell


if __name__ == '__main__':
    dict = LoadData()
    #print(pd.__version__)
    print(type(dict))
    print(FindTouchPosition(DATA_FOLDER + '\\fingernail_test_passive_setup2\\touch_test_fingernail_passive_setup2_place_A1_center_v2.csv'))

