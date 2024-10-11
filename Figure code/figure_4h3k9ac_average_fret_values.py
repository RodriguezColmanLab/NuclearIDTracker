import numpy
from matplotlib import pyplot as plt
from scipy.stats import linregress

from organoid_tracker.imaging import list_io
import lib_figures

_TRACKING_INPUT_FILE = "../../Data/H3K9ac reporter/Tracking data.autlist"
_FRET_SIGNAL_KEY = "intensity"


def main():
    for experiment in list_io.load_experiment_list_file(_TRACKING_INPUT_FILE):
        timings = experiment.images.timings()

        times_h = list()
        fret_values = list()

        for time_point in experiment.time_points():
            total_fret_value = 0
            total_nuclei = 0
            for position in experiment.positions.of_time_point(time_point):
                intensity = experiment.position_data.get_position_data(position, _FRET_SIGNAL_KEY)
                if intensity is not None:
                    total_fret_value += intensity
                    total_nuclei += 1
            if total_nuclei > 0:
                times_h.append(timings.get_time_h_since_start(time_point))
                fret_values.append(total_fret_value / total_nuclei)

        # Convert to numpy
        times_h = numpy.array(times_h)
        fret_values = numpy.array(fret_values)

        # Do a linear fit
        regression = linregress(times_h[times_h > 5], fret_values[times_h > 5])
        regression_values = regression.slope * times_h + regression.intercept

        figure = lib_figures.new_figure()
        ax_unscaled, ax_scaled = figure.subplots(nrows=2, ncols=1, sharex=True)
        ax_unscaled.plot(times_h, fret_values, c="black")
        ax_unscaled.plot(times_h, regression_values, "--", c="black", alpha=0.3, linewidth=6)
        ax_unscaled.set_ylabel("Average FRET signal")
        ax_unscaled.set_title(experiment.name.get_name())

        ax_scaled.plot(times_h, fret_values / regression_values, c="black")
        ax_scaled.set_xlabel("Time (h)")
        ax_scaled.set_ylabel("Scaled FRET signal")
        plt.show()


if __name__ == "__main__":
    main()
