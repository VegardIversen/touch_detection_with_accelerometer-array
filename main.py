"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
from main_scripts.project_thesis import (setup1_results,
                                         setup2_results,
                                         setup3_results,
                                         custom_plots)
from main_scripts.bandpassing_touch import (setup1_predict_reflections)
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    """Run functions from generate_results.py
    NOTE:   Individual figures can be plotted by wrapping it with
            plt.close('all') and plt.show()
    """
    set_fontsizes()

    while input not in ['0', '1', '2', '3']:
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
            from utils.setups import Setup1
            setup = Setup1()
            setup1_predict_reflections(setup)
        else:
            print('Please type 1, 2 or 3 for their respective setups '
                  'or 0 for code currently under development.')
        plt.show()
    return 0


if __name__ == '__main__':
    main()
