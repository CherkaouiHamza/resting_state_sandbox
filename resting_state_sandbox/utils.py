""" Simple module to store usefull function.
"""

# Sys import
import os

# Third party import
from nilearn import image


def subsample_nifti(input_filename, output_filename=None, idx_start=0):
    """ Select only the scan from input_filename present in index and save it
    in output_filename.

    Parameters:
    -----------
    input_filename: str,
        the filename of the input nifti file.

    output_filename: str,
        the filename to store the subsampled nifti file.

    idx_start: int,
        index of scan to start.

    Return:
    -------
    sub_img: 4d nibabel.nifti1.Nifti1Image,
        the subsampled list of scans in nifti format.
    """
    if output_filename is None:
        basename = os.path.splitext(input_filename)[0]
        output_filename = basename + "_{0}_first_discard.nii".format(idx_start)
    img = image.load_img(input_filename)
    if len(img.shape) != 4:
        raise ValueError("Nifti volum scans should be 4d data,"
                         " got {0} dim".format(len(img.shape)))
    index = range(idx_start, img.shape[-1])
    sub_img = image.concat_imgs([image.index_img(img, idx) for idx in index])
    sub_img.to_filename(output_filename)
    return sub_img
