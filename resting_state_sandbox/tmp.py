# discaring of 10 first scans

func_data_filename = os.path.join(data_dir, func_data_filename)
sub_img = index_img(load_img(func_data_filename), slice(idx_start, None))
sub_img.to_filename(os.path.join(data_dir,"func.nii"))

###############################################################################
# wm and csf masking

# create and save binary mask for csf and wm
csf_filename = os.path.join("pypreprocess_output", "mwc2T1.nii.gz")
wm_filename = os.path.join("pypreprocess_output", "mwc3T1.nii.gz")
anat_csf_img = load_img(csf_filename)
anat_wm_img = load_img(wm_filename)
anat_csf = (np.nan_to_num(anat_csf_img.get_data()) > 0).astype(int)
anat_wm = (np.nan_to_num(anat_wm_img.get_data()) > 0).astype(int)
anat_csf_img = nibabel.nifti1.Nifti1Image(anat_csf, affine=fmri_img.affine)
anat_wm_img = nibabel.nifti1.Nifti1Image(anat_wm, affine=fmri_img.affine)
binmask_csf_filename = os.path.join("pypreprocess_output", "bmwc2T1.nii.gz")
binmask_wm_filename = os.path.join("pypreprocess_output", "bmwc3T1.nii.gz")
anat_csf_img.to_filename(binmask_csf_filename)
anat_wm_img.to_filename(binmask_wm_filename)

# get and flatten the wm and csf average time serie
csf_4d_data = apply_mask(subject_data.func[0], binmask_csf_filename)
wm_4d_data = apply_mask(subject_data.func[0], binmask_wm_filename)
csf_4d_flatten = NiftiMasker().fit_transform(csf_4d_data)
wm_4d_flatten = NiftiMasker().fit_transform(wm_4d_data)
dirt_data = np.vstack([wm_flatten, csf_flatten])

###############################################################################
# wm and csf masking

# extract wm and csf average time series
print("Extracting the Cerebrospinal fluid (CSF)...")
csf_filename = os.path.join("pypreprocess_output", "mwc2T1.nii.gz")
csf_time_serie = NiftiMasker(mask_img=csf_filename).fit_transform(func_fname)
print("Extracting the white matter (WM)...")
wm_filename = os.path.join("pypreprocess_output", "mwc3T1.nii.gz")
wm_time_serie = NiftiMasker(mask_img=wm_filename).fit_transform(func_fname)

dirty_data = np.vstack([csf_time_serie, wm_time_serie])

###############################################################################
# make design matrix of PC components

# orthogonalizing the dirty regressor
print("Computing the PCA out of the CSF and the WM...")
pca = PCA(n_components=2)
pca.fit(dirty_data)
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
print("Defining the cleaning design matrix..")
design_matrix = make_design_matrix(frametimes, hrf_model='spm',
                                   add_regs=pca.components_,
                                   add_reg_names=[["csfwm_pc1", "csfwm_pc2"]])

###############################################################################
# cleaning GLM

# fit a first GLM to clean the data
print("Fitting the first GLM to clean the data...")
cleaner = FirstLevelModel(t_r=tr, slice_time_ref=0.5, noise_model='ar1',
                           standardize=False)
cleaner.fit(run_imgs=fmri_img, design_matrices=design_matrix)
dirty_fmri_img = cleaner.results_.predict(design_matrix)
print("Clean the data...")
fmri_img -= dirty_fmri_img
