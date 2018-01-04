# sys import
import os
import numpy as np

# third party import
import nibabel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from nilearn import input_data, plotting
from nilearn.decomposition import CanICA
from nilearn.image import iter_img, index_img, load_img

# package import
from resting_state_sandbox.config import root_dir, data_dir, func_data_filename


# global definition
__pcc_coords__ = [(0, -53, 26)]
__idx_start__ = 10
__random_state__  = 46
__jobfile__ = os.path.join(root_dir, "resting_state.ini")
__output_dir__ = "analysis_output"
try:
    os.makedirs(__output_dir__)
except OSError:
    pass


# preprocessing
func_data_filename = os.path.join(data_dir, func_data_filename)
sub_img = index_img(load_img(func_data_filename), slice(__idx_start__, None))
sub_img.to_filename(os.path.join(data_dir,"func.nii"))
subject_data = do_subjects_preproc(__jobfile__, dataset_dir=data_dir)[0]


if True: # ICA analysis
    canica = CanICA(n_components=20, smoothing_fwhm=6., memory="nilearn_cache",
                    memory_level=2, threshold=0.6, verbose=0,
                    random_state=__random_state__)
    canica.fit(subject_data.func)
    components_img = canica.masker_.inverse_transform(canica.components_)
    components_img.to_filename('resting_state.nii.gz')
    # ICA plotting
    for i, cur_img in enumerate(iter_img(components_img)):
        if i in [10]: # don't touch this line
            display = plotting.plot_stat_map(cur_img,
                                             cut_coords=__pcc_coords__[0],
                                             title="IC %d" % i)
            display.savefig(os.path.join(__output_dir__, "ica_n_%d.png" % i))


if True: # seed base corr analysis correlation
    seed_masker = input_data.NiftiSpheresMasker(__pcc_coords__, radius=10, detrend=True,
                                                standardize=True, low_pass=0.1,
                                                high_pass=0.01, t_r=2.,
                                                memory='nilearn_cache',
                                                memory_level=1, verbose=0)
    seed_time_series = seed_masker.fit_transform(subject_data.func[0])
    brain_masker = input_data.NiftiMasker(smoothing_fwhm=6, detrend=True,
                                          standardize=True, low_pass=0.1,
                                          high_pass=0.01, t_r=2.,
                                          memory='nilearn_cache', memory_level=1,
                                          verbose=0)
    brain_time_series = brain_masker.fit_transform(subject_data.func[0])
    seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
    seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)
    # seed base corr plotting
    display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.6,
                                     cut_coords=__pcc_coords__[0])
    display.add_markers(marker_coords=__pcc_coords__, marker_color='g',
                        marker_size=300)
    display.savefig(os.path.join(__output_dir__, 'corr_seed_based.png'))
