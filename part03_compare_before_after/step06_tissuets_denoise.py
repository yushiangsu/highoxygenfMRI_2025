# %%
import os
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, math_img, get_data
from nilearn.datasets import load_mni152_brain_mask

from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cmc
# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
outdir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")

subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
tr = 2
# %%
# get mask
group_outdir = os.path.join(outdir, 'group')
group_mask = os.path.join(group_outdir, 'group_mask.nii.gz')

mask_data = get_data(group_mask)
# %% 
# get atlas
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
        #f'(img > {th_val}) & (mask_img > 99/100)', 
        f'(img > 0.8) & (mask_img > 99/100)', 
        img = tiss_tpm_resample, mask_img = group_mask)
    
    #tiss_mask['tiss_name'] = tiss_th_mask
    tiss_dat[tiss_name] = get_data(tiss_th_mask)

# %%
all_ts = dict({
    'gm': [],
    'wm': [],
    'csf': [],
    'arte': [],
    'vein': []
})

for i_sub in subinfo['participant_id']:
    subid = i_sub
    #subid = 'sub-001'
    print(f'Working on {subid}!')
    sub_datdir = os.path.join(datdir, f'{subid}')
    
    for i_run in [1, 2]:
        # skip when subject == 010 and run == 1
        # because the switch delayed 10 seconds
        if (i_sub == 'sub-010') and (i_run == 1):
            continue
        
        # Load data
        img_path = os.path.join(
            sub_datdir,
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-denoise_boldpsc.nii.gz')
        
        img_data = get_data(img_path)
        
        # get ts
        for tiss_name in all_ts.keys():
            tiss_tss = img_data[np.where(tiss_dat[tiss_name])]
            all_ts[tiss_name].append(np.mean(tiss_tss, axis = 0))
            
# %% stack in to matrix
for tiss_name in all_ts.keys():
    all_ts[tiss_name] = np.stack(all_ts[tiss_name], axis = 0)

# %%
# define bootstrap funciton
def boot_func(idx, tiss_name):
    return np.mean(all_ts[tiss_name][idx, :], axis = 0)

# run bootstrapping
boot_tss = dict()
boot_meants = dict()
for tiss_name in all_ts.keys():
    boot_tss[tiss_name] = bootstrap(
        (np.arange(all_ts[tiss_name].shape[0]),),
        statistic = lambda idx: boot_func(idx = idx, tiss_name = tiss_name),
        n_resamples = 10000,
        confidence_level = 0.95,
        axis = 0)
    boot_meants[tiss_name] = boot_tss[tiss_name].bootstrap_distribution.mean(axis = 1)
    
# orgnaize into data frame
plt_dat = pd.DataFrame(
    boot_meants, index = np.arange(0*2, 600*2, 2))

# %%
fig, axs = plt.subplots(1, figsize = (12, 3))
sns.lineplot(plt_dat[['vein', 'arte', 'csf', 'gm', 'wm', ]][5:], 
             dashes = False, ax = axs,
             palette="cmc.batlowS")
axs.axvline(x = 300, color = 'grey', linestyle = 'dashed')
axs.axvline(x = 600, color = 'grey', linestyle = 'dashed')
axs.set_xlabel('Time (minutes)')
axs.set_ylabel('Signal Changes (%)')
axs.set_xticks(np.arange(0, 1200 + 1, 300))
axs.set_xticklabels(["0", "5", "10", "15", "20"])
fig.savefig("bold_ts_afterdenoise.pdf")

# %%
