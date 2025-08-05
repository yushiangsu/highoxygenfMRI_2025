# %%
import os
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds

# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fmriprep/")
outdir = os.path.join(prjdir, "derivatives/melodic_ica/")

# ICA FIX will based on original native space
# FIX will transform native space to FSL MNI space to identify componenets.
# Then, apply the nuisance regession on native space.
spacename = 'T1w'

subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
# %%
for i_sub in subinfo['participant_id']:
    sub_outdir = os.path.join(outdir, f'{i_sub}')
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)
    for i_run in [1, 2]:      
        with open("./melodic_icafix_template.fsf", "r") as f:
            fsf = f.read()
            fsf = fsf.replace('sub-001', f"{i_sub}")
            fsf = fsf.replace('run-1', f"run-{i_run}")
            fsf = fsf.replace('space-T1w', f"space-{spacename}")
            
            with open(os.path.join(sub_outdir, f"{i_sub}_run-{i_run}_space-{spacename}_melodicica_design.fsf"), "w") as out_f:
                out_f.write(fsf)
            
            # get motion parameter
            bold_img = os.path.join(
                datdir, f"{i_sub}", "func",
                f"{i_sub}_task-rest_run-{i_run}_space-{spacename}_desc-preproc_bold.nii.gz")
            conf_pd = load_confounds(bold_img, strategy = ["motion"], motion = "basic")[0]
                
            with open(os.path.join(sub_outdir, f"run-{i_run}_motion.par"), "w") as mc_f:
                for i_r in range(conf_pd.shape[0]):
                    for i_c in range(conf_pd.shape[1]):
                        mc_f.write(f"{conf_pd.iloc[i_r, i_c]}")
                        if (i_c+1) != conf_pd.shape[1]:
                            mc_f.write("  ")
                        else:
                            mc_f.write("\n")

# %%
