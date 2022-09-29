print(__file__)
from data_processing.cross_correlation_position import FindTouchPosition 
from data_viz_files.display_table_grid import draw
from pathlib import Path

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'

if __name__=='__main__':
    cell = FindTouchPosition(DATA_FOLDER+ '\\first_test_touch_active_setup2_5\\touch_test_active_setup2_5_A1_v2.csv')
    draw(cell)