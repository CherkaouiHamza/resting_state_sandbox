# sys import
import os
import time

# third party import
import numpy as np
import nibabel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from nilearn import plotting as niplt
from nilearn.input_data import NiftiSpheresMasker, NiftiMasker
from nilearn.decomposition import CanICA
from nilearn.image import iter_img, index_img, load_img
from nistats.design_matrix import (make_design_matrix,
                                   check_design_matrix)
from nistats.first_level_model import FirstLevelModel
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report

# package import
from resting_state_sandbox.config import root_dir, data_dir, func_data_filename

#########################################################################
# Global definition
pcc_coords = (0, -53, 26)
idx_start = 10
jobfile = os.path.join(root_dir, "preprocessing.ini")
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#########################################################################
# Preprocessing
#func_data_filename = os.path.join(data_dir, func_data_filename)
#sub_img = index_img(load_img(func_data_filename), slice(idx_start, None))
#sub_img.to_filename(os.path.join(data_dir,"func.nii"))

subject_data = do_subjects_preproc(jobfile, dataset_dir=data_dir)[0]
fmri_img = load_img(subject_data.func[0])

#########################################################################
# Seed base correlation analysis
corr_output_dir = os.path.join(output_dir, "seed_base_corr")
if not os.path.exists(corr_output_dir):
    os.makedirs(corr_output_dir)
seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                            standardize=True, low_pass=0.1,
                                            high_pass=0.01, t_r=2.,
                                            memory='nilearn_cache',
                                            memory_level=1, verbose=0)
seed_time_series = seed_masker.fit_transform(subject_data.func[0])
brain_masker = NiftiMasker(smoothing_fwhm=6, detrend=True,
                                      standardize=True, low_pass=0.1,
                                      high_pass=0.01, t_r=2.,
                                      memory='nilearn_cache', memory_level=1,
                                      verbose=10)
brain_time_series = brain_masker.fit_transform(fmri_img)
seed_based_corr = np.dot(brain_time_series.T,
                                 seed_time_series) / seed_time_series.shape[0]
seed_based_corr_img = brain_masker.inverse_transform(seed_based_corr.T)
display = niplt.plot_stat_map(seed_based_corr_img, threshold=0.6,
                              cut_coords=pcc_coords)
display.add_markers(marker_coords=[pcc_coords], marker_color='g',
                    marker_size=300)
display.savefig(os.path.join(corr_output_dir, 'corr_seed_based.png'))
seed_based_corr_img.to_filename(os.path.join(corr_output_dir,
