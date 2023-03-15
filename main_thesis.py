from data_processing.plate_properties import teflon_plate
import results_thesis as r



def main():
    #teflon_plate()
    #r.results_setup1()
    #r.data_viz('scaleogram', 'plate10mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    #r.velocities()
    #r.wave_type_plots()
    r.data_viz('ssq', 'plate10mm\\setup1\\touch', 'nik_touch_v1', channel='channel 1')
    
if __name__ == '__main__':
    main()