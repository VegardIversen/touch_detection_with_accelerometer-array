"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np
from main_scripts.project_thesis import (setup1_results,
                                         setup2_results,
                                         setup3_results,
                                         custom_plots)
from main_scripts.bandpassing_touch import (setup1_predict_reflections)
from main_scripts.generate_ideal_signal import (generate_ideal_signal,
                                                compare_to_ideal_signal)
from utils.data_visualization.visualize_data import compare_signals, set_fontsizes
from utils.global_constants import SAMPLE_RATE
from utils.plate_setups import Setup_3x3


def main():
    """Run functions from generate_results.py
    NOTE:   Individual figures can be plotted by wrapping it with
            plt.close('all') and plt.show()
    """
    set_fontsizes()

    print('\nWhich setup do you want to generate results for?')
    print('1: Setup 1')
    print('2: Setup 2')
    print('3: Setup 3')
    print('0: Dev')
    # user_input = input('Enter number: ')
    user_input = '0'
    if user_input == '1':
        setup1_results()
    elif user_input == '2':
        setup2_results()
    elif user_input == '3':
        setup3_results()
    elif user_input == '0':
        SETUP = Setup_3x3()
        SETUP.draw()
        ideal_signal = generate_ideal_signal(setup=SETUP,
                                             propagation_speed_mps=600 ,
                                             attenuation_dBpm=10,
                                             time_end=0.125,
                                             frequency_start=1,
                                             frequency_stop=6500)
        TIME_AXIS = np.linspace(0, 0.2, ideal_signal.shape[0])
        fig, axs = plt.subplots(3, 3, squeeze=False,
                                sharex=True, sharey=True)
        for i in range(3):
            for j in range(3):
                sensor_number = str(i * 3 + j + 1)
                axs[i, j].plot(
                    TIME_AXIS, ideal_signal['Sensor ' + sensor_number])
                axs[i, j].grid()
                axs[i, j].set_title('Sensor ' + sensor_number)

    else:
        print('Please type 1, 2 or 3 for their respective setups '
              'or 0 for code currently under development.')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
