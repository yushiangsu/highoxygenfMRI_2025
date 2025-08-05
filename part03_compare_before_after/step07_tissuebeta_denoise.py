# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from nilearn.image import math_img, resample_to_img, get_data
from pingouin import bayesfactor_pearson
import seaborn as sns
import matplotlib.pyplot as plt
# %%
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
betadir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")

subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
# %%
# get mask
group_outdir = os.path.join(betadir, 'group')
group_mask = os.path.join(group_outdir, 'group_mask.nii.gz')
mask_data = get_data(group_mask)

# %% get atlas
tiss_info = {
    'gm': {
        'rawimg': os.path.join('../atlas/MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz'),
        'npick': 10000,
    },
    'wm': {
        'rawimg': os.path.join('../atlas/MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-WM_probseg.nii.gz'),
        'npick': 10000,
    },
    'csf': {
        'rawimg': os.path.join('../atlas/MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-CSF_probseg.nii.gz'),
        'npick': 500,
    },
    'arte': {
        'rawimg': os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_ToF_Thresh.nii.gz'),
        'npick': 500,
    },
    'vein': {
        'rawimg': os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_swi_Thresh.nii.gz'),
        'npick': 500,
    },
}
tiss_dat = dict()
for tiss_name, tiss_dict in tiss_info.items():
    tiss_tpm_resample = resample_to_img(
        source_img = tiss_dict['rawimg'],
        target_img = group_mask,
        interpolation = 'continuous'
    )
    
    th_val = np.sort(
        get_data(tiss_tpm_resample)[mask_data == 1])[-tiss_dict['npick']]
    print(f'{tiss_name} threshold at p = {th_val}')
    
    tiss_th_mask = math_img(
        f'(img > {th_val}) & (mask_img > 99/100)', 
        img = tiss_tpm_resample, mask_img = group_mask)
    
    tiss_dat[tiss_name] = get_data(tiss_th_mask)

# %%
p2p1_vals = dict()
for tiss_name in tiss_dat.keys():
    p2p1_vals[tiss_name] = list()

for i_sub in subinfo['participant_id']:
    subid = i_sub
    #subid = 'sub-001'
    print(f'Working on {subid}!')
    
    sub_betadir = os.path.join(betadir, f'{subid}')
    p2p1_beta = get_data(
        os.path.join(sub_betadir, f'{subid}_space-MNI152NLin2009cAsym_desc-p2p1diff_beta.nii.gz'))
    for tiss_name in tiss_dat.keys():
        
        p2p1_vals[tiss_name].append(np.mean(p2p1_beta[tiss_dat[tiss_name] == 1]))
p2p1_dat = pd.DataFrame(p2p1_vals)

# %%
# load physio data
physio_dat = pd.read_csv("../part01_analyse_physio/physio_all.csv")
physio_p2p1_dat = pd.concat([
    physio_dat.loc[physio_dat['onset'] == 300].iloc[:, 0:1].reset_index(drop = True),
    physio_dat.loc[physio_dat['onset'] == 300].iloc[:, 3:].reset_index(drop = True) - 
    physio_dat.loc[physio_dat['onset'] == 0].iloc[:, 3:].reset_index(drop = True) ],
                            axis = 1).groupby('subid').mean().reset_index(drop = True)

comb_p2p1_dat = pd.concat([physio_p2p1_dat, p2p1_dat], axis = 1)

# %%
tiss_list = ['vein', 'arte']
physio_list = ['puls_HR', 'resp_RVT']
fig, axs = plt.subplots(2, 2, figsize = (6, 6))

for i_physio in range(len(physio_list)):
    for i_tiss in range(len(tiss_list)):
        sns.regplot(
            data = comb_p2p1_dat, 
            x = tiss_list[i_tiss],  
            y = physio_list[i_physio], 
            ax = axs[i_physio, i_tiss])
        na_idx = np.logical_or(
            np.isnan(comb_p2p1_dat[tiss_list[i_tiss]]),
            np.isnan(comb_p2p1_dat[physio_list[i_physio]]))
        rhor, pval = pearsonr(
            comb_p2p1_dat[physio_list[i_physio]][~na_idx], 
            comb_p2p1_dat[tiss_list[i_tiss]][~na_idx])
        axs[i_physio, i_tiss].set_title(f'r={rhor:.3f}; p={pval:.3f}')

plt.tight_layout()
fig.savefig("bold_tiss_physio_denoise.pdf")
# %%
# calculate bayesfactor for vein ~ puls_HR
na_idx = np.logical_or(
    np.isnan(comb_p2p1_dat['vein']),
    np.isnan(comb_p2p1_dat['puls_HR']))
rhor, pval = pearsonr(
    comb_p2p1_dat['puls_HR'][~na_idx], 
    comb_p2p1_dat['vein'][~na_idx])
bayesfactor_pearson(rhor, n = np.sum(~na_idx))

# %%
# calculate bayesfactor for arte ~ puls_HR
na_idx = np.logical_or(
    np.isnan(comb_p2p1_dat['arte']),
    np.isnan(comb_p2p1_dat['puls_HR']))
rhor, pval = pearsonr(
    comb_p2p1_dat['puls_HR'][~na_idx], 
    comb_p2p1_dat['arte'][~na_idx])
bayesfactor_pearson(rhor, n = np.sum(~na_idx))
# %%
