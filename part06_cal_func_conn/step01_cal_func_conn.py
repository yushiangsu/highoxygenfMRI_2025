# %%
import os
import re
import pandas as pd
import numpy as np
import scipy
from nilearn.image import load_img, index_img, clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_schaefer_2018

# %% Settings
p1_idx = slice( 30, 150)
p2_idx = slice(180, 300)
p3_idx = slice(330, 450)
p4_idx = slice(480, 600)

# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
betadir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")

# %%
# get mask
group_betadir = os.path.join(betadir, 'group')
group_glmmask = os.path.join(group_betadir, 'group_glmmask.nii.gz')

# %%
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")

outdir = os.path.join(prjdir, "derivatives/nilearn_funcconn/")
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# %% prepare Schaefer atlas/parcellations
atlas_schaefer = fetch_atlas_schaefer_2018(n_rois = 100, yeo_networks = 7)
schaefer_masker = NiftiLabelsMasker(
        labels_img = atlas_schaefer['maps'],
        labels = atlas_schaefer['labels'],
        mask_img = group_glmmask,
        standardize = False,
    )
schaefer_labels = [
    _.decode('utf-8') for _ in atlas_schaefer['labels'] ]
schaefer_networks = [
    re.match(r'^7Networks_(LH|RH)_(.+?)_(.+?)', _).group(2) 
    for _ in schaefer_labels ]

# %%
all_corrmat_p1 = list()
all_corrmat_p2 = list()
all_corrmat_p3 = list()
all_corrmat_p4 = list()
for i_sub in subinfo["participant_id"]:
    subid = i_sub
    #subid = 'sub-001'
    print(f"Processing {subid} now!")
    subdir = os.path.join(datdir, f'{subid}')
    sub_outdir = os.path.join(outdir, f'{subid}')
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)
    sub_inputs = {
        'bold_p1': [],
        'bold_p2': [],
        'bold_p3': [],
        'bold_p4': [],
        'mask': [],
    }
    for i_run in [1, 2]:
        # get bold images
        bold_img = os.path.join(
            subdir,
            f'{subid}_task-rest_run-{i_run}_space-MNI152NLin2009cAsym_desc-denoise_bold.nii.gz'
        )
        
        # load images and denoise
        den_img_p1 = clean_img(index_img(bold_img, p1_idx),
            detrend = True, standardize = "zscore_sample",
            low_pass = 1/12, high_pass = 1/128,
            t_r = 2)
        den_img_p2 = clean_img(index_img(bold_img, p2_idx),
            detrend = True, standardize = "zscore_sample",
            low_pass = 1/12, high_pass = 1/128,
            t_r = 2)
        den_img_p3 = clean_img(index_img(bold_img, p3_idx),
            detrend = True, standardize = "zscore_sample",
            low_pass = 1/12, high_pass = 1/128,
            t_r = 2)
        den_img_p4 = clean_img(index_img(bold_img, p4_idx),
            detrend = True, standardize = "zscore_sample",
            low_pass = 1/12, high_pass = 1/128,
            t_r = 2)
        
        # 
        sub_inputs['bold_p1'].append(den_img_p1)
        sub_inputs['bold_p2'].append(den_img_p2)
        sub_inputs['bold_p3'].append(den_img_p3)
        sub_inputs['bold_p4'].append(den_img_p4)
        
        # get mask images
        sub_inputs['mask'].append(load_img(os.path.join(
            subdir,
            f'{subid}_task-rest_run-{i_run}_space-MNI152NLin2009cAsym_desc-denoise_mask.nii.gz'
        )))
    
    sub_ts_p1 = list()
    sub_ts_p2 = list()
    sub_ts_p3 = list()
    sub_ts_p4 = list()
    for i_run in range(2):
        sub_ts_p1.append(schaefer_masker.fit_transform(
            sub_inputs['bold_p1'][i_run]))
        sub_ts_p2.append(schaefer_masker.fit_transform(
            sub_inputs['bold_p2'][i_run]))
        sub_ts_p3.append(schaefer_masker.fit_transform(
            sub_inputs['bold_p3'][i_run]))
        sub_ts_p4.append(schaefer_masker.fit_transform(
            sub_inputs['bold_p4'][i_run]))
    
    corrmeasure = ConnectivityMeasure(
        kind = "correlation",
        standardize = 'zscore_sample',
        discard_diagonal = True,
        vectorize = False
    )
    corrmat_p1 = corrmeasure.fit_transform(sub_ts_p1)
    corrmat_p2 = corrmeasure.fit_transform(sub_ts_p2)
    corrmat_p3 = corrmeasure.fit_transform(sub_ts_p3)
    corrmat_p4 = corrmeasure.fit_transform(sub_ts_p4)
    for i_run in range(2):
        scipy.io.savemat(
            os.path.join(sub_outdir, f'{subid}_task-rest_run-{i_run+1}_conn.mat'),
            {'p1': corrmat_p1[i_run, :, :].astype('double'),
             'p2': corrmat_p2[i_run, :, :].astype('double'),
             'p3': corrmat_p3[i_run, :, :].astype('double'),
             'p4': corrmat_p4[i_run, :, :].astype('double')}
            )

    all_corrmat_p1.append(corrmat_p1.mean(axis = 0))
    all_corrmat_p2.append(corrmat_p2.mean(axis = 0))
    all_corrmat_p3.append(corrmat_p3.mean(axis = 0))
    all_corrmat_p4.append(corrmat_p4.mean(axis = 0))

# %%
#
# output to matlab nps
#
group_outdir = os.path.join(outdir, 'group')
if not os.path.exists(group_outdir):
    os.makedirs(group_outdir)

# specify design of how to do statistic 
scipy.io.savemat(
    os.path.join(group_outdir, 'pairedt_des.mat'),
    {'desmat': np.concatenate(
         [np.concatenate([np.eye(22), np.ones((22, 1))* 1], axis = 1), 
          np.concatenate([np.eye(22), np.ones((22, 1))*-1], axis = 1)], 
         axis = 0)})
scipy.io.savemat(
    os.path.join(group_outdir, 'pairedt_con.mat'),
    {'contrast': np.concatenate([np.zeros(22), np.array([1])])})
scipy.io.savemat(
    os.path.join(group_outdir, 'pairedt_excblk.mat'),
    {'excblk': np.concatenate([np.arange(1, 22+1), np.arange(1, 22+1)])})

# save group p2p1 matrix
scipy.io.savemat(
    os.path.join(group_outdir, 'p2p1corr.mat'),
    {'corrmat': np.stack(all_corrmat_p2 + all_corrmat_p1, axis = -1)}
)

# save group p3p1 matrix
scipy.io.savemat(
    os.path.join(group_outdir, 'p3p1corr.mat'),
    {'corrmat': np.stack(all_corrmat_p3 + all_corrmat_p1, axis = -1)}
)

# save group p4p1 matrix
scipy.io.savemat(
    os.path.join(group_outdir, 'p4p1corr.mat'),
    {'corrmat': np.stack(all_corrmat_p4 + all_corrmat_p1, axis = -1)}
)

# save rois info
schaefer_roiinfo = pd.read_csv(
    os.path.join(
        group_outdir, 
        'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'))
schaefer_roiinfo['ROI Name'].to_csv(
    os.path.join(group_outdir, 'roi_label.txt'), sep = "\t", header = False, index = False)
schaefer_roiinfo[['R', 'A', 'S']].to_csv(
    os.path.join(group_outdir, 'roi_goc.txt'), sep = "\t", header = False, index = False)