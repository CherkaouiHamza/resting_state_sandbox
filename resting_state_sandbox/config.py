import os
import ConfigParser

# [don't edit this file]

root_dir = os.path.dirname(os.path.realpath(__file__))

config = ConfigParser.RawConfigParser()
config.read('config.ini')
items = dict(config.items('GLOBAL'))
data_home = items['data_home']
subject_id = int(items['subject_id'])


if subject_id == 2:
    subj_dir = os.path.join("20161130_e00521_REGLAGE02_REGLAGE02", "nii")
    data_dir = os.path.join(data_home, subj_dir)
    func_data_filename = "s010a1001_epi3p4mm_rest.nii"
    anat_data_filename = "T1.nii"
elif subject_id == 5:
    subj_dir = os.path.join("20170727_e00730_REGLAGE05_REGLAGE05", "nii")
    data_dir = os.path.join(data_home, subj_dir)
    func_data_filename = "s005a1001_epi3p4mm_rest.nii"
    anat_data_filename = "T1.nii"
elif subject_id == 6:
    subj_dir = os.path.join("20171114_e00766_REGLAGE06_REGLAGE06", "nii")
    data_dir = os.path.join(data_home, subj_dir)
    func_data_filename = "s006a1001_epi3p4mm_rest.nii"
    anat_data_filename = "T1.nii"
elif subject_id == 7:
    subj_dir = os.path.join("20171207_e00787_REGLAGE07_REGLAGE07", "nii")
    data_dir = os.path.join(data_home, subj_dir)
    func_data_filename = "s004a1001_epi3p4mm_rest.nii"
    anat_data_filename = "T1.nii"
elif subject_id == 8:
    subj_dir = os.path.join("20171208_e00788_REGLAGE08_REGLAGE08", "nii")
    data_dir = os.path.join(data_home, subj_dir)
    func_data_filename = "s004a1001_epi3p4mm_rest.nii"
    anat_data_filename = "T1.nii"
else:
    raise(ValueError("Wrong subject_id, got {0},"
                     "please check 'config.ini'".format(subject_id)))
