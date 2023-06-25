from data_processing.plate_properties import teflon_plate
import results_thesis as r


def main():
    # teflon_plate()
    # r.results_setup1()
    # r.data_viz('scaleogram', 'plate10mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    # r.velocities()
    # r.wave_type_plots()
    # r.data_viz('ssq', 'plate20mm\\setup1_vegard\\touch', 'touch_v1', channel='channel 1')
    # r.simulated_data_vel()
    # r.comsol_data200()
    # r.comsol_data200_phase_diff()
    # r.show_A0_and_S0_wave()
    # r.comsol_data50()
    # r.un_reassigned_spectrogram()
    # r.show_A0_and_S0_wave_comsol_diagonal(20, save=True, size=0.45, expected_arrival_time=False)
    # r.warping_map()
    # r.dispersion_compensation_Wilcox(pertubation=True,alpha=0.2)
    # r.dispersion_compensation_Wilcox(position=25)
    # r.wave_number_to_omega()
    # r.all_calculate_phase_velocities(0.75, save=False)
    # r.velocites_modes()
    # r.draw_simulated_plate()
    # r.testing_wilcox_disp()
    # r.test_dispersion_compensation_gen_signals()
    # r.pressure_wave_oscilloscope(save=True)
    # r.comsol_pulse()
    # r.gen_pulse_dispersion(distance=6, save=True, size=0.45)
    # r.theory_disperson_curve()
    # r.find_combinations_for_velocity_range()
    # print(r.calculate_velocity(0.97357*1e9, 910, 0.468908))
    # r.draw_all_plates()
    # r.touch_signal_plot()
    # r.touch_signal_plot_hold()
    # r.swipe_signal()
    # r.chirp_signal()
    # r.comsol_wave_prop_all()
    # r.COMSOL_velocity_curve(
    #     save=True,
    #     name="real",
    #     filenum=5,
    #     y_max=2.5,
    #     number_modes=2,
    #     freq_max=45,
    #     size=0.45,
    # )
    # r.find_best_values()
    # r.REAL_plate_velocities(save=True, size=0.45)
    # r.print_material_values()
    # r.COMSOL_dispersion(
    #     position=10, theoretical=True, reflections=True, save=True, size=0.45
    # )
    r.REAL_dispersion(save=True, size=0.45, distance=0.2, reflections=True)
    # r.REAL_wavemodes(save=True, size=0.33, chirp=True)
    pass


if __name__ == "__main__":
    main()
