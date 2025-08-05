# %%
import os
import pandas as pd
import nipype.pipeline.engine as pe
from nipype.interfaces import ants
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util

# %% prepare data path
prjdir = "/work/O2Resting/"

# %% get participants list
subjinfo = pd.read_csv(
    os.path.join(prjdir, 'rawdata', 'participants.tsv'),
    sep = '\t')
subjlist = subjinfo.participant_id.to_list()
# %%
def file_renamer(subj):
    subs_list = [
        (f"bold_r1/_subj_{subj}/"
         f"{subj}_task-rest_run-1_space-T1w_desc-denoise_bold_trans.nii.gz",
         f"{subj}/"
         f"{subj}_task-rest_run-1_space-MNI152NLin2009cAsym_desc-denoise_bold.nii.gz"),
        (f"bold_r2/_subj_{subj}/"
         f"{subj}_task-rest_run-2_space-T1w_desc-denoise_bold_trans.nii.gz",
         f"{subj}/"
         f"{subj}_task-rest_run-2_space-MNI152NLin2009cAsym_desc-denoise_bold.nii.gz"),
        (f"mask_r1/_subj_{subj}/"
         f"{subj}_task-rest_run-1_space-T1w_desc-denoise_mask_trans.nii.gz",
         f"{subj}/"
         f"{subj}_task-rest_run-1_space-MNI152NLin2009cAsym_desc-denoise_mask.nii.gz"),
        (f"mask_r2/_subj_{subj}/"
         f"{subj}_task-rest_run-2_space-T1w_desc-denoise_mask_trans.nii.gz",
         f"{subj}/"
         f"{subj}_task-rest_run-2_space-MNI152NLin2009cAsym_desc-denoise_mask.nii.gz"),
        ]
    return subs_list

def wrapinlist(strpath):
    return [strpath]

# %% get inputs
getinputs = pe.Node(util.IdentityInterface(
    fields = ["subj"]), name = "getinputs")

img_templates = {
    "ref_img_r1": os.path.join(
        "derivatives", "fmriprep", "{subj}", "func", 
        "{subj}_task-rest_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz"),
    "ref_img_r2": os.path.join(
        "derivatives", "fmriprep", "{subj}", "func", 
        "{subj}_task-rest_run-2_space-MNI152NLin2009cAsym_boldref.nii.gz"),
    "tran_file": os.path.join(
        "derivatives", "fmriprep", "{subj}", "anat", 
        "{subj}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"),
    "inputimg_r1": os.path.join(
        "derivatives", "fix_agg_originalcomps", "{subj}",
        "{subj}_task-rest_run-1_space-T1w_desc-denoise_bold.nii.gz"),
    "inputimg_r2": os.path.join(
        "derivatives", "fix_agg_originalcomps", "{subj}",
        "{subj}_task-rest_run-2_space-T1w_desc-denoise_bold.nii.gz"),
    "inputmask_r1": os.path.join(
        "derivatives", "fix_agg_originalcomps", "{subj}",
        "{subj}_task-rest_run-1_space-T1w_desc-denoise_mask.nii.gz"),
    "inputmask_r2": os.path.join(
        "derivatives", "fix_agg_originalcomps", "{subj}",
        "{subj}_task-rest_run-2_space-T1w_desc-denoise_mask.nii.gz"),
}
pickfiles = pe.Node(nio.SelectFiles(
    img_templates, base_directory = prjdir, sort_filelist = True),
                    name = "pickfiles")

# %% do transformation
antsatr1 = pe.Node(
    ants.ApplyTransforms(
        input_image_type = 3,
        float = True,
        interpolation = "Linear",
        invert_transform_flags = [False]),
    name = "ants_applytransforms_run1")
antsatr2 = pe.Node(
    ants.ApplyTransforms(
        input_image_type = 3,
        float = True,
        interpolation = "Linear",
        invert_transform_flags = [False]),
    name = "ants_applytransforms_run2")
antsatr1_mask = pe.Node(
    ants.ApplyTransforms(
        float = True,
        interpolation = "Linear",
        invert_transform_flags = [False]),
    name = "ants_applytransforms_run1_mask")
antsatr2_mask = pe.Node(
    ants.ApplyTransforms(
        float = True,
        interpolation = "Linear",
        invert_transform_flags = [False]),
    name = "ants_applytransforms_run2_mask")
path2list = pe.Node(util.Function(
    input_names = ["strpath"],
    output_names = ["strpath_in_list"],
    function = wrapinlist),
                    name = "path2list")
# %% data sink
outputnames = pe.Node(
    util.Function(input_names = ["subj"],
                output_names = ["subs_list"],
                function = file_renamer),
    name = "outputnames")
# %%
savefiles = pe.Node(nio.DataSink(
    base_directory = os.path.join(prjdir, "derivatives"),
    container = "fix_agg_originalcomps"),
                    name = "savefiles")
# %%
at2pwf = pe.Workflow(
    name=f"ants_applytransforms_bold",
    base_dir = os.path.join(prjdir, "temporary", "nipype_work"))

at2pwf.connect([
    (getinputs, pickfiles, [("subj", "subj")]),
    (getinputs, outputnames, [("subj", "subj")]),
    (pickfiles, path2list, [("tran_file", "strpath")]),
    (pickfiles, antsatr1, [("ref_img_r1", "reference_image"), ("inputimg_r1", "input_image")]),
    (path2list, antsatr1, [("strpath_in_list", "transforms")]),
    (pickfiles, antsatr2, [("ref_img_r2", "reference_image"), ("inputimg_r2", "input_image")]),
    (path2list, antsatr2, [("strpath_in_list", "transforms")]),
    (pickfiles, antsatr1_mask, [("ref_img_r2", "reference_image"), ("inputmask_r1", "input_image")]),
    (path2list, antsatr1_mask, [("strpath_in_list", "transforms")]),
    (pickfiles, antsatr2_mask, [("ref_img_r2", "reference_image"), ("inputmask_r2", "input_image")]),
    (path2list, antsatr2_mask, [("strpath_in_list", "transforms")]),
    (antsatr1, savefiles, [("output_image", "bold_r1")]),
    (antsatr2, savefiles, [("output_image", "bold_r2")]),
    (antsatr1_mask, savefiles, [("output_image", "mask_r1")]),
    (antsatr2_mask, savefiles, [("output_image", "mask_r2")]),
    (outputnames, savefiles, [("subs_list", "substitutions")])
])
getinputs.iterables = [
    #("subj", ['sub-001']),
    ("subj", subjlist)
]
# %% RUNNNNNNN
results = at2pwf.run()
# %%
