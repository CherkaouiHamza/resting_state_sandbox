# sys import
import sys
import os
import numpy as np

# third party import
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from nilearn import input_data
from nilearn.decomposition import CanICA
from nilearn import plotting
from nilearn.image import iter_img

# package import
from resting_state_sandbox.config import root_dir, claire5_data_dir
from resting_state_sandbox.utils import subsample_nifti


#------------------------------------------------------------------------------

# set the path to data and config file
dataset_dir = claire5_data_dir
jobfile = os.path.join(root_dir, "example", "claire5", "claire5_resting_state.ini")
output_dir = "analysis_output"
try:
    os.makedirs(output_dir)
except OSError:
    pass

#------------------------------------------------------------------------------

# preprocessing

data_filename = os.path.join(dataset_dir, "s005a1001_epi3p4mm.nii")
subsample_nifti(data_filename, idx_start=10)
results = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)[0]
func_filename = results.func

#------------------------------------------------------------------------------

# ICA analysis
# Don't change random_state=46
canica = CanICA(n_components=20, smoothing_fwhm=6., memory="nilearn_cache",
                memory_level=2, threshold=0.6, verbose=0, random_state=46)
canica.fit(func_filename)
components_img = canica.masker_.inverse_transform(canica.components_)
components_img.to_filename('nicolas2_resting_state.nii.gz')

# ICA plotting
for i, cur_img in enumerate(iter_img(components_img)):
    if i in [14]: # the closest to the DMN
        display = plotting.plot_stat_map(cur_img, cut_coords=(0, -52, 24),
                                         title="IC %d" % i,)
display.savefig(os.path.join(output_dir, 'claire5_ica.png'))

#------------------------------------------------------------------------------

# seed base analysis
pcc_coords = [(0, -52, 18)]
seed_masker = input_data.NiftiSpheresMasker(pcc_coords, radius=12, detrend=True,
                                            standardize=True, low_pass=0.1,
                                            high_pass=0.01, t_r=2.,
                                            memory='nilearn_cache',
                                            memory_level=1, verbose=0)
seed_time_series = seed_masker.fit_transform(func_filename[0])
brain_masker = input_data.NiftiMasker(smoothing_fwhm=10, detrend=True,
                                      standardize=True, low_pass=0.1,
                                      high_pass=0.01, t_r=2.,
                                      memory='nilearn_cache', memory_level=1,
                                      verbose=0)
brain_time_series = brain_masker.fit_transform(func_filename[0])
seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)

# seed base plotting
display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.7,
                                 cut_coords=(0, -52, 24))
display.add_markers(marker_coords=pcc_coords, marker_color='g',
                    marker_size=300)
display.savefig(os.path.join(output_dir, 'claire5_seed_based.png'))
