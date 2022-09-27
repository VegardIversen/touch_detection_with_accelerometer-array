print(__file__)
from data_processing.cross_correlation_position import FindTouchPosition 
from data_viz_files.display_table_grid import draw
from pathlib import Path
DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
if __name__=='__main__':
    cell = FindTouchPosition(DATA_FOLDER+ '\\fingernail_test_passive_setup2\\touch_test_fingernail_passive_setup2_place_A1_center_v2.csv')
    draw(cell)