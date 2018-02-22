# sys import
import os
import time

# third party import
import numpy as np
from nilearn import plotting as niplt
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiSpheresMasker, NiftiMasker
from nilearn.image import  index_img, load_img, high_variance_confounds
from nistats.design_matrix import make_design_matrix
from nistats.first_level_model import FirstLevelModel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report

# package import
from resting_state_sandbox.config import root_dir, data_dir, func_data_filename

#########################################################################
# Global definition
tr = 2.
pcc_coords = (0, -53, 26)
idx_start = 10
jobfile = os.path.join(root_dir, "preprocessing.ini")
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#########################################################################
# Preprocessing

print("I will start the preprocessing...")
subject_data = do_subjects_preproc(jobfile, dataset_dir=data_dir)[0]
func_fname = subject_data.func[0]
fmri_img = load_img(func_fname)
n_scans = fmri_img.shape[-1]

#########################################################################
# Seed base GLM analysis

# manage outputs
glm_output_dir = os.path.join(output_dir, "seed_based_glm")
if not os.path.exists(glm_output_dir):
    os.makedirs(glm_output_dir)
stats_start_time = time.ctime()

# extract high variance components
print("Extract high variance components...")
n_confounds = 5
hv_comps = high_variance_confounds(func_fname, n_confounds)

# wm and csf masking
print("Extracting the Cerebrospinal fluid (CSF)...")
csf_filename = os.path.join("pypreprocess_output", "mwc2T1.nii.gz")
csf_time_serie = NiftiMasker(mask_img=csf_filename).fit_transform(func_fname)
print("Extracting the white matter (WM)...")
wm_filename = os.path.join("pypreprocess_output", "mwc3T1.nii.gz")
wm_time_serie = NiftiMasker(mask_img=wm_filename).fit_transform(func_fname)

dirty_data = np.vstack([csf_time_serie, wm_time_serie])

# make design matrix of PC components
print("Computing the PCA out of the CSF and the WM...")
pca = PCA(n_components=2)
pca.fit(dirty_data)
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
print("Defining the cleaning design matrix..")
design_matrix = make_design_matrix(frametimes, hrf_model='spm',
                                   add_regs=pca.components_,
                                   add_reg_names=[["csfwm_pc1", "csfwm_pc2"]])

# fit a first GLM to clean the data
print("Fitting the first GLM to clean the data...")
cleaner = FirstLevelModel(t_r=tr, slice_time_ref=0.5, noise_model='ar1',
                           standardize=False)
cleaner.fit(run_imgs=fmri_img, design_matrices=design_matrix)
dirty_fmri_img = cleaner.results_.predict(design_matrix)
print("Clean the data...")
fmri_img -= dirty_fmri_img

#########################################################################
# Cleaning the data

# extract the seed
print("Extracting the average seed time serie...")
seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                 standardize=True, low_pass=0.1,
                                 high_pass=0.01, t_r=2.,
                                 memory='nilearn_cache',
                                 memory_level=1, verbose=1)
seed_time_series = seed_masker.fit_transform(func_fname)

# define the design matrix
print("Defining the main design matrix..")
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
add_reg_names = ["hv_comp_{0}".format(i) for i in range(n_confounds)]
add_reg_names.insert(0, "seed")
regressors = np.hstack([seed_time_series, hv_comps])
design_matrix = make_design_matrix(frametimes, hrf_model='spm',
                                   add_regs=regressors,
                                   add_reg_names=add_reg_names)

# define contrast
print("Defining the seed contrast")
dmn_contrast = np.array([1] + [0]*(design_matrix.shape[1]-1))
contrasts = {'seed_based_glm': dmn_contrast}

# fit the GLM
print("Fitting the second GLM to extract the network of the seed...")
glm = FirstLevelModel(t_r=tr, slice_time_ref=0.5, noise_model='ar1',
                      standardize=False)
glm.fit(run_imgs=fmri_img, design_matrices=design_matrix)

# compute z_map for the given contrast
print("Computing the contrast...")
z_map = glm.compute_contrast(contrasts['seed_based_glm'], output_type='z_score')

#########################################################################
# Saving output

map_path = os.path.join(glm_output_dir, 'seed_based_glm.nii.gz')
print("Saving the zmap...")
z_map.to_filename(map_path)
display = niplt.plot_stat_map(z_map, threshold=3.0,
                              cut_coords=pcc_coords)
display.add_markers(marker_coords=[pcc_coords], marker_color='g',
                    marker_size=300)
display.savefig(os.path.join(glm_output_dir, 'seed_based_glm.png'))
stats_report_filename = os.path.join(subject_data.reports_output_dir,
                                     "report_stats.html")
print("Creating the subject html report...")
generate_subject_stats_report(stats_report_filename,
                              contrasts, {'seed_based_glm': map_path},
                              glm.masker_.mask_img_,
                              design_matrices=[design_matrix],
                              subject_id=subject_data.subject_id,
                              anat=load_img(subject_data.anat),
                              display_mode='ortho',
                              threshold=3., cluster_th=50,
                              start_time=stats_start_time, TR=tr,
                              nscans=n_scans, frametimes=frametimes)
