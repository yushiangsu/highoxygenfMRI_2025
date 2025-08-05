# %%
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.image import math_img, resample_to_img, get_data
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.ndimage import binary_dilation
from scipy.stats import pearsonr
from pingouin import bayesfactor_pearson
# %%
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
betadir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")

# %%
# get mask
group_betadir = os.path.join(betadir, 'group')
group_mask = os.path.join(group_betadir, 'group_mask.nii.gz')
group_glmmask = os.path.join(group_betadir, 'group_glmmask.nii.gz')
mask_data = get_data(group_mask)

# %% get atlas
tiss_info = {
    'arte': {
        'rawimg': os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_ToF_Thresh.nii.gz'),
    },
    'vein': {
        'rawimg': os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_swi_Thresh.nii.gz'),
    },
}

tiss_tpm = dict()
for tiss_name, tiss_dict in tiss_info.items():
    tiss_tpm_resample = resample_to_img(
        source_img = tiss_dict['rawimg'],
        target_img = group_mask,
        interpolation = 'continuous'
    )
    tiss_tpm[tiss_name] = get_data(tiss_tpm_resample)

# %% 
# get beta maps
p2p1_beta = os.path.join(group_betadir, 'p2p1', 'group_p2p1_beta.nii.gz')
p2p1_data = np.squeeze(get_data(p2p1_beta))

# %% get rois
n_rois = 100
schaefer_atlas = fetch_atlas_schaefer_2018(
    n_rois = n_rois, yeo_networks =  7, resolution_mm =1) # 1mm
schaefer_labels = [_.decode('utf-8') for _ in schaefer_atlas['labels']]
schaefer_network = np.array(
    [ re.match(r'^7Networks_(LH|RH)_(.+?)_(.+?)', _).group(2) 
     for _ in schaefer_labels ])
schaefer_dat = get_data(resample_to_img(
    source_img = schaefer_atlas['maps'],
    target_img = group_mask,
    interpolation = 'nearest'))
schaefer_mask = schaefer_dat.copy()
schaefer_mask[get_data(group_glmmask) == 0] = 0

# %%
for i in range(n_rois + 1):
    print(f'{i}: {np.sum(schaefer_mask == i)}')
# %%
p2p1_list = list()
for i in np.arange(1, n_rois + 1, 1):
    if np.sum(schaefer_mask == i) < 20:
        p2p1_list.append(np.nan)
    else:
        p2p1_list.append(np.mean(p2p1_data[schaefer_mask == i]))

# %% rois dilation (assumed smoothness to 6 mm)
dilation_maps = list()
tpm_info = dict({
    'label': [], 'network': []})
for tiss_name in tiss_tpm.keys():
    tpm_info[tiss_name] = list()
    
for i_roi in range(len(schaefer_labels)):
    # get roi, do dilation, turn to image
    tpm_info['label'].append(schaefer_labels[i_roi])
    tpm_info['network'].append(schaefer_network[i_roi])
    pickroi_idx = i_roi + 1
    pickroi_dat = np.zeros_like(schaefer_dat)
    pickroi_dat[schaefer_dat == pickroi_idx] = 1.0
    pickroi_dat = binary_dilation(pickroi_dat, iterations = 3)
    pickroi_dat[mask_data == 0] = 0.0 # remove the dialated voxel that is not brain
    pickroi_dat[(schaefer_dat != pickroi_idx) & (schaefer_dat != 0)] = 0.0
    
    # calculate occupancy of vessel, wm, csf
    for tiss_name in tiss_tpm.keys():
        tpm_info[tiss_name].append(tiss_tpm[tiss_name][pickroi_dat == 1.0].mean())

tpm_info['p2p1_psc'] = p2p1_list
tpm_pd = pd.DataFrame(tpm_info)
# %%
fig, axs = plt.subplots(1, 2, figsize = (6, 3))
sns.regplot(data = tpm_pd, x = 'vein', y = 'p2p1_psc', ax = axs[0])
sns.regplot(data = tpm_pd, x = 'arte', y = 'p2p1_psc', ax = axs[1])
for idx, vname in enumerate(['vein', 'arte']):
    rhor, pval = pearsonr(tpm_info['p2p1_psc'], tpm_info[f'{vname}'])
    axs[idx].set_title(f'r={rhor:.3f}; p={pval:.3f}')
plt.tight_layout()
fig.savefig("./bold_roi_beta_cor_tiss_denoise.pdf")

# %%
# calculate bayesfactor for vein ~ p2p1_psc
rhor, pval = pearsonr(
    tpm_pd['p2p1_psc'], 
    tpm_pd['vein'])
bayesfactor_pearson(rhor, n = tpm_pd.shape[0])
# %%
# calculate bayesfactor for vein ~ p2p1_psc
rhor, pval = pearsonr(
    tpm_pd['p2p1_psc'], 
    tpm_pd['arte'])
bayesfactor_pearson(rhor, n = tpm_pd.shape[0])
# %%
