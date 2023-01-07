"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
from generate_results import (setup3_2_results,
                              setup7_results,
                              setup9_results,
                              custom_plots)
from data_visualization.visualize_data import set_fontsizes


def main():
    """Run results_setup*n() functions from generate_results.py
    NOTE:   Individual figures can be plotted by wrapping it with
            plt.close('all') and plt.show()
    """
    set_fontsizes()

    setup = ''
    while input not in ['3', '7', '9', '0']:
        print('\nWhich setup do you want to generate results for?')
        print('3: Setup 3.2')
        print('7: Setup 7')
        print('9: Setup 9')
        print('0: Custom plots')
        setup = input('Enter number: ')
        if setup == '3':
            setup3_2_results()
        elif setup == '7':
            setup7_results()
        elif setup == '9':
            setup9_results()
        elif setup == '0':
            custom_plots()
        else:
            print('Please type 3, 7 or 9 for their respective setups '
                  'or c for custom plots used in the report.')

        plt.show()
    return 0


if __name__ == '__main__':
    main()
