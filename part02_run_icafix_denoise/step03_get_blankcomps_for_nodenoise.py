# %%
import os
import re
import pandas as pd
import numpy as np

# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fmriprep/")
outdir = os.path.join(prjdir, "derivatives/melodic_ica/")
vesdir = os.path.join(prjdir, "derivatives/ants_tpmatsub/")

# %%
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
# %%
# write new text files
for i_sub in subinfo['participant_id']:

    for i_run in [1, 2]:
        icomp_blankout = list()
        with open(
            os.path.join(
                outdir, f'{i_sub}', f'space-T1w_run-{i_run}.ica', 
                'fix4melview_Standard_thr20.txt'), 
            'r') as f:
            for txt_line in f.readlines():
                if re.match(r'^\[.+', txt_line):
                    icomp_blankout.append(
                        '[]\n')
                else:
                    txt_re = re.match(r'^([0-9]+), (.+), (.+)\n', txt_line)
                    if txt_re:
                        if txt_re.group(3) == 'True':
                            icomp_blankout.append(
                                f"{txt_re.group(1)}, Signal, False\n")
                        else:
                            icomp_blankout.append(txt_line)
                    else:
                        icomp_blankout.append(txt_line)
        with open(
            os.path.join(
                outdir, f'{i_sub}', f'space-T1w_run-{i_run}.ica', 
                'blankfix.txt'), 
            'w') as outf:
            outf.writelines(icomp_blankout)