from typing import Optional, List, Tuple

from pyopenms import *


def load_data(
        mgf_filename: str,
        range_spectrum: tuple = (0, 1600),  # spectrum range
        min_intensity: Optional[float] = None,  # determine threshold to remove
        mz_binning_width: float = 5.0,
) -> Tuple[List[float], List[float]]:
    """
    Process and return date from xml file

    :param mgf_filename: path to file in string format
    :param range_spectrum: defined range of m/z
    :param  min_intensity: minimum intensity to select file
    :param mz_binning_width: choose m/z width for merging spectra

    :return: m/z and intensity values list
    """
    min_mz, max_mz = range_spectrum
    experiment = MSExperiment()
    result_experiment = MSExperiment()
    MzMLFile().load(mgf_filename, experiment)

    for spectrum in experiment:
        filtered_mz, filtered_intensity = [], []
        mz_exp, intensity_exp = spectrum.get_peaks()
        if min_intensity is None:
            min_intensity = max(intensity_exp) / 2000
        for mz, intensity in zip(mz_exp, intensity_exp):
            if min_mz <= mz <= max_mz:
                filtered_mz.append(mz)
                if intensity >= min_intensity:
                    filtered_intensity.append(intensity)
                else:
                    filtered_intensity.append(0)
        spectrum.set_peaks((filtered_mz, filtered_intensity))

        gf = SavitzkyGolayFilter()
        gf_params = gf.getDefaults()
        gf_params.setValue(b'frame_length', 5, b'')
        gf.setParameters(gf_params)
        gf.filter(spectrum)
        result_experiment.addSpectrum(spectrum)

    # select spectrum from experiment in defined range
    ms_levels = []
    for ex in result_experiment:
        lvl = ex.getMSLevel()
        if lvl not in ms_levels:
            ms_levels.append(lvl)

    # Merge various MS levels
    sd = SpectraMerger()
    p = Param()
    p.setValue('block_method:rt_block_size', result_experiment.getNrSpectra())
    p.setValue('mz_binning_width', mz_binning_width)
    p.setValue('block_method:ms_levels', ms_levels)
    sd.setParameters(p)
    sd.mergeSpectraBlockWise(result_experiment)

    return result_experiment[0].get_peaks()

