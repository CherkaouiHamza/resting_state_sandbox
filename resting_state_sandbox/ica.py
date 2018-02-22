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
random_state  = 46
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
# ICA analysis
ica_output_dir = os.path.join(output_dir, "ica")
if not os.path.exists(ica_output_dir):
    os.makedirs(ica_output_dir)
canica = CanICA(n_components=20, smoothing_fwhm=6., memory="nilearn_cache",
                memory_level=2, threshold=0.6, verbose=10,
                random_state=random_state)
canica.fit(fmri_img)
components_img = canica.masker_.inverse_transform(canica.components_)
for i, cur_img in enumerate(iter_img(components_img)):
        display = niplt.plot_stat_map(cur_img,
                                      cut_coords=pcc_coords,
                                      title="IC {0}".format(i))
        display.savefig(os.path.join(ica_output_dir,
                                     "ica_n_{0}.png".format(i)))
        cur_img.to_filename(os.path.join(ica_output_dir,
                                         "ica_n_{0}.nii.gz".format(i)))
