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
jobfile = os.path.join(root_dir, "resting_state.ini")
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#########################################################################
# Preprocessing
if False:
    func_data_filename = os.path.join(data_dir, func_data_filename)
    sub_img = index_img(load_img(func_data_filename), slice(idx_start, None))
    sub_img.to_filename(os.path.join(data_dir,"func.nii"))

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
seed_based_correlation_img.to_filename(os.path.join(corr_output_dir,
                                       "corr_seed_based.nii.gz"))

#########################################################################
# Seed base GLM analysis
glm_output_dir = os.path.join(output_dir, "seed_based_glm")
if not os.path.exists(glm_output_dir):
    os.makedirs(glm_output_dir)
stats_start_time = time.ctime()
seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                 standardize=True, low_pass=0.1,
                                 high_pass=0.01, t_r=2.,
                                 memory='nilearn_cache',
                                 memory_level=1, verbose=0)
seed_time_series = seed_masker.fit_transform(subject_data.func[0])
tr = 2.
n_scans = fmri_img.shape[-1]
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
design_matrix = make_design_matrix(frametimes, hrf_model='spm',
                                   add_regs=seed_time_series,
                                   add_reg_names=["pcc"])
dmn_contrast = np.array([1] + [0]*(design_matrix.shape[1]-1))
contrasts = {'seed_based_glm': dmn_contrast}
fmri_glm = FirstLevelModel(t_r=tr, slice_time_ref=0.5, noise_model='ar1',
                           standardize=False)
fmri_glm.fit(run_imgs=fmri_img, design_matrices=design_matrix)
anat_img = load_img(subject_data.anat)
z_map = fmri_glm.compute_contrast(contrasts['seed_based_glm'], output_type='z_score')
map_path = os.path.join(glm_output_dir, 'seed_based_glm.nii.gz')
nibabel.save(z_map, map_path)
display = niplt.plot_stat_map(z_map, threshold=3.0,
                              cut_coords=pcc_coords)
display.add_markers(marker_coords=[pcc_coords], marker_color='g',
                    marker_size=300)
display.savefig(os.path.join(glm_output_dir, 'seed_based_glm.png'))
stats_report_filename = os.path.join(subject_data.reports_output_dir,
                                     "report_stats.html")
generate_subject_stats_report(stats_report_filename,
                              contrasts, {'seed_based_glm': map_path},
                              fmri_glm.masker_.mask_img_,
                              design_matrices=[design_matrix],
                              subject_id=subject_data.subject_id,
                              anat=anat_img, display_mode='ortho',
                              threshold=3., cluster_th=50,
                              start_time=stats_start_time, TR=tr,
                              nscans=n_scans, frametimes=frametimes)
