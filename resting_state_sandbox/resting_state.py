# sys import
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# third party import
import nibabel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.external.nistats.design_matrix import (make_design_matrix,
                                                         check_design_matrix,
                                                         plot_design_matrix)
from pypreprocess.external.nistats.glm import FirstLevelGLM
from nilearn import input_data
from nilearn.decomposition import CanICA
from nilearn import plotting
from nilearn.image import iter_img, index_img, load_img

# package import
from resting_state_sandbox.config import root_dir, data_dir, data_filename


#------------------------------------------------------------------------------

__pcc_coords__ = [(0, -53, 26)]
__idx_start__ = 10
__random_state__  = 46
__jobfile__ = os.path.join(root_dir, "resting_state.ini")
__output_dir__ = "analysis_output"
try:
    os.makedirs(__output_dir__)
except OSError:
    pass

#------------------------------------------------------------------------------

# preprocessing
# discard __idx_start__ first scans
__idx_start__ = 10
data_filename = os.path.join(data_dir, data_filename)
sub_img = index_img(load_img(data_filename), slice(__idx_start__, None))
new_data_filename = (os.path.splitext(data_filename)[0]
                     + "_{0}_first_discard.nii".format(__idx_start__))
sub_img.to_filename(new_data_filename)
# do the preprocessing
results = do_subjects_preproc(__jobfile__, dataset_dir=data_dir)[0]
func_filename = results.func

#------------------------------------------------------------------------------

# ICA analysis
canica = CanICA(n_components=20, smoothing_fwhm=6., memory="nilearn_cache",
                memory_level=2, threshold=0.6, verbose=0, random_state=__random_state__)
canica.fit(func_filename)
components_img = canica.masker_.inverse_transform(canica.components_)
components_img.to_filename('resting_state.nii.gz')

# ICA plotting
for i, cur_img in enumerate(iter_img(components_img)):
    display = plotting.plot_stat_map(cur_img, cut_coords=__pcc_coords__,
                                     title="IC %d" % i,)
display.savefig(os.path.join(__output_dir__, 'ica.png'))

#------------------------------------------------------------------------------

if False:
    # seed base analysis correlation
    seed_masker = input_data.NiftiSpheresMasker(__pcc_coords__, radius=10, detrend=True,
                                                standardize=True, low_pass=0.1,
                                                high_pass=0.01, t_r=2.,
                                                memory='nilearn_cache',
                                                memory_level=1, verbose=0)
    seed_time_series = seed_masker.fit_transform(func_filename[0])
    brain_masker = input_data.NiftiMasker(smoothing_fwhm=6, detrend=True,
                                          standardize=True, low_pass=0.1,
                                          high_pass=0.01, t_r=2.,
                                          memory='nilearn_cache', memory_level=1,
                                          verbose=0)
    brain_time_series = brain_masker.fit_transform(func_filename[0])
    seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
    seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
    seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)

    # seed base plotting
    display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.65,
                                     cut_coords=__pcc_coords__)
    display.add_markers(marker_coords=__pcc_coords__, marker_color='g',
                        marker_size=300)
    display.savefig(os.path.join(__output_dir__, 'seed_based.png'))

#------------------------------------------------------------------------------

if False:
    # seed base analysis GLM
    # construct experimental paradigm
    stats_start_time = time.ctime()
    tr = 7. #!
    n_scans = 96 #!
    _duration = 6 #!
    n_conditions = 2 #!
    epoch_duration = _duration * tr
    conditions = ['rest', 'active'] * 8 #!
    duration = epoch_duration * np.ones(len(conditions))
    onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                        len(conditions))
    paradigm = pd.DataFrame(
        {'onset': onset, 'duration': duration, 'name': conditions})

    hfcut = 2 * 2 * epoch_duration

    with open(sd.func[0].split(".")[0] + "_onset.txt", "w") as fd:
        for c, o, d in zip(conditions, onset, duration):
            fd.write("%s %s %s\r\n" % (c, o, d))

    # preprocess the data
    subject_data = do_subjects_preproc(__jobfile__, dataset_dir=data_dir)[0]

    # construct design matrix
    nscans = len(subject_data.func[0]) #! n_scans
    frametimes = np.linspace(0, (nscans - 1) * tr, nscans)
    drift_model = 'Cosine'
    hrf_model = 'spm + derivative'
    design_matrix = make_design_matrix(
        frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model,
        period_cut=hfcut)

    # plot and save design matrix
    ax = plot_design_matrix(design_matrix)
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    dmat_outfile = os.path.join(subject_data.output_dir, 'design_matrix.png')
    plt.savefig(dmat_outfile, bbox_inches="tight", dpi=200)

    # specify contrasts
    contrasts = {}
    _, matrix, names = check_design_matrix(design_matrix)
    contrast_matrix = np.eye(len(names))
    for i in range(len(names)):
        contrasts[names[i]] = contrast_matrix[i]

    # more interesting contrasts"""
    contrasts = {'active-rest': contrasts['active'] - contrasts['rest']} #!

    # fit GLM
    print('\r\nFitting a GLM (this takes time) ..')
    fmri_glm = FirstLevelGLM(noise_model='ar1', standardize=False).fit(
        [nibabel.concat_images(subject_data.func[0])], design_matrix)


    # save computed mask
    mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz") #!
    print("Saving mask image %s" % mask_path) #!
    nibabel.save(fmri_glm.masker_.mask_img_, mask_path) #!

    # compute bg unto which activation will be projected
    anat_img = nibabel.load(subject_data.anat)

    print("Computing contrasts ..")
    z_maps = {}
    effects_maps = {}
    for contrast_id, contrast_val in contrasts.items():
        print("\tcontrast id: %s" % contrast_id)
        z_map, t_map, eff_map, var_map = fmri_glm.transform(
            contrasts[contrast_id], contrast_name=contrast_id, output_z=True,
            output_stat=True, output_effects=True, output_variance=True)

        # store stat maps to disk
        for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                                  [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(
                subject_data.output_dir, '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if contrast_id == 'active-rest' and dtype == "z":
                z_maps[contrast_id] = map_path

            print("\t\t%s map: %s" % (dtype, map_path))

        print

    # do stats report
    stats_report_filename = os.path.join(subject_data.reports_output_dir,
                                         "report_stats.html")
    contrasts = dict((contrast_id, contrasts[contrast_id])
                     for contrast_id in z_maps.keys())
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.masker_.mask_img_,
        design_matrices=[design_matrix],
        subject_id=subject_data.subject_id,
        anat=anat_img,
        display_mode='ortho',
        threshold=3.,
        cluster_th=50,  # 'large' clusters
        start_time=stats_start_time,
        paradigm=paradigm,
        TR=tr,
        nscans=nscans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model)

    print("\r\nStatistic report written to %s\r\n" % stats_report_filename)
