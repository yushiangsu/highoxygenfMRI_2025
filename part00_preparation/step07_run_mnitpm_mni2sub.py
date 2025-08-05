# %%
import os
import pandas as pd
import nipype.pipeline.engine as pe
from nipype.interfaces import ants
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util

# %% prepare data path
prjdir = "/work/O2Resting/"
rawdir = os.path.join(prjdir, "derivatives/fmriprep/")
outdir = os.path.join(prjdir, "derivatives/ants_tpmatsub/")
if not os.path.exists(outdir):
    os.makedirs(outdir)

# specify 'gm' or 'wm' or 'csf'
target_name = 'gm'          
# get vascular atlas
target_tpm = dict({
    'gm': os.path.join(
        prjdir, 'atlas', 
        'MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz'),
    'wm': os.path.join(
        prjdir, 'atlas', 
        'MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-WM_probseg.nii.gz'),
    'csf': os.path.join(
        prjdir, 'atlas', 
        'MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-CSF_probseg.nii.gz'),
})

# %% get participants list
subjinfo = pd.read_csv(
    os.path.join(prjdir, 'rawdata', 'participants.tsv'),
    sep = '\t')
subjlist = subjinfo.participant_id.to_list()
# %%
def gm_renamer(subj):
    subs_list = [(
        f"_subj_{subj}/"
        f"tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg_trans.nii.gz",
        f"{subj}/"
        f"{subj}_space-T1w_label-GM_probseg.nii.gz")]
    return subs_list
def wm_renamer(subj):
    subs_list = [(
        f"_subj_{subj}/"
        f"tpl-MNI152NLin2009cAsym_res-01_label-WM_probseg_trans.nii.gz",
        f"{subj}/"
        f"{subj}_space-T1w_label-WM_probseg.nii.gz")]
    return subs_list
def csf_renamer(subj):
    subs_list = [(
        f"_subj_{subj}/"
        f"tpl-MNI152NLin2009cAsym_res-01_label-CSF_probseg_trans.nii.gz",
        f"{subj}/"
        f"{subj}_space-T1w_label-CSF_probseg.nii.gz")]
    return subs_list

def wrapinlist(strpath):
    return [strpath]

# %% get inputs
getinputs = pe.Node(util.IdentityInterface(
    fields = ["subj"]), name = "getinputs")

img_templates = {
    "ref_img": os.path.join(
        "derivatives", "fmriprep", "{subj}", "anat", 
        "{subj}_desc-preproc_T1w.nii.gz"),
    "tran_file": os.path.join(
        "derivatives", "fmriprep", "{subj}", "anat", 
        "{subj}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"),
}
pickfiles = pe.Node(nio.SelectFiles(
    img_templates, base_directory = prjdir, sort_filelist = True),
                    name = "pickfiles")

# %% do transformation
antsat2p = pe.Node(
    ants.ApplyTransforms(
        dimension = 3, input_image = target_tpm[f'{target_name}'],
        float = True,
        interpolation = "LanczosWindowedSinc",
        invert_transform_flags = [False]),
    name = "ants_applytransforms")

path2list = pe.Node(util.Function(
    input_names = ["strpath"],
    output_names = ["strpath_in_list"],
    function = wrapinlist),
                    name = "path2list")
# %% data sink
if target_name == 'gm':
    outputnames = pe.Node(
        util.Function(input_names = ["subj"],
                    output_names = ["subs_list"],
                    function = gm_renamer),
        name = "outputnames")
elif target_name == 'wm':
    outputnames = pe.Node(
        util.Function(input_names = ["subj"],
                    output_names = ["subs_list"],
                    function = wm_renamer),
        name = "outputnames")
else:
    outputnames = pe.Node(
        util.Function(input_names = ["subj"],
                    output_names = ["subs_list"],
                    function = csf_renamer),
        name = "outputnames")
# %%
savefiles = pe.Node(nio.DataSink(
    base_directory = os.path.join(prjdir, "derivatives"),
    container = "ants_tpmatsub"),
                    name = "savefiles")
# %%
at2pwf = pe.Workflow(
    name=f"ants_applytransforms_{target_name}",
    base_dir = os.path.join(prjdir, "temporary", "nipype_work"))

at2pwf.connect([
    (getinputs, pickfiles, [("subj", "subj")]),
    (getinputs, outputnames, [("subj", "subj")]),
    (pickfiles, path2list, [("tran_file", "strpath")]),
    (pickfiles, antsat2p, [("ref_img", "reference_image")]),
    (path2list, antsat2p, [("strpath_in_list", "transforms")]),
    (antsat2p, savefiles, [("output_image", f'{target_name}')]),
    (outputnames, savefiles, [("subs_list", "substitutions")])
])
getinputs.iterables = [
    #("subj", ['sub-001']),
    ("subj", subjlist)
]
# %% RUNNNNNNN
results = at2pwf.run()
# %%