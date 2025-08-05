# %%
import os
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, math_img, get_data
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from nitime.algorithms import get_spectra, multi_taper_psd
from scipy.stats import bootstrap
from scipy.stats import ttest_rel
# %%
def mtm_psd(tss, Fs, NW = 4, low_bias = True, sides = 'onesided',
                 scale_ts = False):
    # tss: series x time (time should be at axis: -1)
    na_idx = np.sum(np.diff(tss), axis = 1) == 0
    tss = tss[~na_idx, :]
    
    # if normalized
    if scale_ts:
        tss = (tss - np.mean(tss, axis = -1, keepdims = True)
               ) / np.std(tss, axis = -1, keepdims = True)
    
    # get estimate of power spectral density using multitaper method
    freq, psd, nu = multi_taper_psd(
        tss,
        Fs = Fs, NW = NW, low_bias = low_bias, sides = sides,
        adaptive = False, jackknife = False)
    
    # return
    return freq, psd, nu

# %% helpful function to create design matrix
total_tr = 600
tr = 2

tps_idx = {
    'p1': np.arange(  0, 150),
    'p2': np.arange(150, 300),
    'p3': np.arange(300, 450),
    'p4': np.arange(450, 600)
}

# %% prepare data path
prjdir = "/work/O2Resting/"
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")

b_datdir = os.path.join(prjdir, "derivatives/fix_agg_blankcomps/")
b_group_mask = os.path.join(
    prjdir, "derivatives/nilearn_univglm_blankcomps_psc/", 
    'group', 'group_mask.nii.gz')
a_datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
a_group_mask = os.path.join(
    prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/", 
    'group', 'group_mask.nii.gz')

# %% 
gm_path = os.path.join(
            '../atlas/MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz')
gm_mask = math_img(
        f'(img > 0.8) & (mask_img > 99/100)', 
        img = resample_to_img(
            source_img = gm_path,
            target_img = b_group_mask,
            interpolation = 'continuous'), 
        mask_img = b_group_mask)
gm_dat = get_data(gm_mask)
# %%
b_psd_dict = dict({
    'p1': [], 'p2': [], 'p3': [], 'p4': []
})
a_psd_dict = dict({
    'p1': [], 'p2': [], 'p3': [], 'p4': []
})
for i_sub in subinfo['participant_id']:
    subid = i_sub
    #subid = 'sub-001'
    print(f'Working on {subid}!')
    b_sub_datdir = os.path.join(b_datdir, f'{subid}')
    a_sub_datdir = os.path.join(a_datdir, f'{subid}')
    b_sub_dict = dict({'p1': [], 'p2': [], 'p3': [], 'p4': []})
    a_sub_dict = dict({'p1': [], 'p2': [], 'p3': [], 'p4': []})
    for i_run in [1, 2]:
        # skip subject == 010 and run == 1
        if (i_sub == 'sub-010') and (i_run == 1):
            continue
        
        # Load data
        b_img_path = os.path.join(
            b_sub_datdir,
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-denoise_boldpsc.nii.gz')
        b_img_data = get_data(b_img_path)
        
        a_img_path = os.path.join(
            a_sub_datdir,
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-denoise_boldpsc.nii.gz')
        a_img_data = get_data(a_img_path)
        
        # get ts
        b_tiss_tss = b_img_data[np.where(gm_dat)]
        a_tiss_tss = a_img_data[np.where(gm_dat)]
        for i_p in ['p1', 'p2', 'p3', 'p4']:
            freq_, b_psds, nu_ = mtm_psd(
                b_tiss_tss[:, tps_idx[i_p]],
                Fs = 1/2, NW = 4, low_bias = True, sides = 'onesided', 
                scale_ts = True
            )
            b_sub_dict[i_p].append(b_psds.mean(axis = 0))
            
            freq_, a_psds, nu_ = mtm_psd(
                a_tiss_tss[:, tps_idx[i_p]],
                Fs = 1/2, NW = 4, low_bias = True, sides = 'onesided', 
                scale_ts = True
            )
            a_sub_dict[i_p].append(a_psds.mean(axis = 0))
    
    for i_p in ['p1', 'p2', 'p3', 'p4']:
        b_psd_dict[i_p].append(np.stack(b_sub_dict[i_p], axis = -1).mean(axis = -1))
        a_psd_dict[i_p].append(np.stack(a_sub_dict[i_p], axis = -1).mean(axis = -1))
            
        
# %% stack in to matrix
for i_p in ['p1', 'p2', 'p3', 'p4']:
    b_psd_dict[i_p] = np.stack(b_psd_dict[i_p], axis = 0)
    a_psd_dict[i_p] = np.stack(a_psd_dict[i_p], axis = 0)
# %%
def boot_func(idx, dat):
    return np.mean(dat[idx, :], axis = 0)
#
b_boot_specs = dict()
b_boot_mean = dict()
a_boot_specs = dict()
a_boot_mean = dict()

for i_p in ['p1', 'p2', 'p3', 'p4']:
    #
    b_boot_specs[i_p] = bootstrap(
        (np.arange(b_psd_dict[i_p].shape[0]),),
        statistic = lambda idx: boot_func(idx = idx, dat = b_psd_dict[i_p]),
        n_resamples = 10000,
        confidence_level = 0.95,
        axis = 0)
    b_boot_mean[i_p] = b_boot_specs[i_p].bootstrap_distribution.mean(axis = 1)
    
    #
    a_boot_specs[i_p] = bootstrap(
        (np.arange(a_psd_dict[i_p].shape[0]),),
        statistic = lambda idx: boot_func(idx = idx, dat = a_psd_dict[i_p]),
        n_resamples = 10000,
        confidence_level = 0.95,
        axis = 0)
    a_boot_mean[i_p] = a_boot_specs[i_p].bootstrap_distribution.mean(axis = 1)
    
# %%
fig, axs = plt.subplots(1, 4, figsize = (10, 3))

b_maxy = np.max(np.concatenate([
        b_boot_mean['p1'], b_boot_mean['p2'], b_boot_mean['p3'], b_boot_mean['p4']]))
a_maxy = np.max(np.concatenate([
        a_boot_mean['p1'], a_boot_mean['p2'], a_boot_mean['p3'], a_boot_mean['p4']]))

for i_idx, i_p in enumerate(['p1', 'p2', 'p3', 'p4']):
    b_ci95 = np.quantile(
        b_boot_specs[i_p].bootstrap_distribution, 
        q = [0.025, 0.975], axis = 1)
    a_ci95 = np.quantile(
        a_boot_specs[i_p].bootstrap_distribution, 
        q = [0.025, 0.975], axis = 1)
    
    axs[i_idx].plot(freq_, b_boot_mean[i_p], color = 'blue', label = 'before')
    axs[i_idx].fill_between(freq_, b_ci95[0, :], b_ci95[1, :], color = 'blue', alpha = 0.2)
    axs[i_idx].plot(freq_, a_boot_mean[i_p], color = 'green', label = 'after')
    axs[i_idx].fill_between(freq_, a_ci95[0, :], a_ci95[1, :], color = 'green', alpha = 0.2)
    axs[i_idx].axhline(y = b_maxy, color = 'blue', alpha = 0.5, linestyle = "dashed")
    axs[i_idx].axhline(y = a_maxy, color = 'green', alpha = 0.5, linestyle = "dashed")
    axs[i_idx].axvspan(0.0, 0.01,color = 'gray', alpha = 0.2)
    axs[i_idx].set_title(f'{i_p}')
    axs[i_idx].set_ylim([0, b_maxy + b_maxy/5])
    axs[i_idx].set_xlabel("Frequency")
    axs[i_idx].set_ylabel("Power Spectral Density")
    axs[i_idx].legend()
    
plt.tight_layout()
fig.savefig("./ica_temporal_spect.pdf")
# %%
plt_bar_dat = pd.DataFrame({
    'b': np.concatenate([
        b_psd_dict['p1'][:, freq_ <= 0.01].mean(axis = 1),
        b_psd_dict['p2'][:, freq_ <= 0.01].mean(axis = 1),
        b_psd_dict['p3'][:, freq_ <= 0.01].mean(axis = 1),
        b_psd_dict['p4'][:, freq_ <= 0.01].mean(axis = 1),]),
    'a': np.concatenate([
        a_psd_dict['p1'][:, freq_ <= 0.01].mean(axis = 1),
        a_psd_dict['p2'][:, freq_ <= 0.01].mean(axis = 1),
        a_psd_dict['p3'][:, freq_ <= 0.01].mean(axis = 1),
        a_psd_dict['p4'][:, freq_ <= 0.01].mean(axis = 1)]),
    'p': np.repeat(["p1", "p2", "p3", "p4"], b_psd_dict['p1'].shape[0])
})
fig, axs = plt.subplots(1, 2, figsize = (3, 3))
sns.barplot(plt_bar_dat, x = 'p', y = 'b', ax = axs[0])
sns.barplot(plt_bar_dat, x = 'p', y = 'a', ax = axs[1])
#axs[0].set_title("Original data at 0.0 - 0.01 Hz")
axs[0].set_ylim(0, 30)
axs[0].set_yticks(np.arange(0, 30, 5))
axs[0].set_ylabel("Power Spectram Density")
#axs[1].set_title("Cleaned data at 0.0 - 0.01 Hz")
axs[1].set_ylim(0, 30)
axs[1].set_yticks(np.arange(0, 30, 5))
axs[1].set_ylabel("")
plt.tight_layout()
fig.savefig("./ica_temporal_spect_bar.pdf")

# %%
# Statistics
# we only care about difference between p2 and p1
ttest_rel(b_psd_dict['p1'][:, freq_ <= 0.01].mean(axis = 1), 
          b_psd_dict['p2'][:, freq_ <= 0.01].mean(axis = 1))
# %%
ttest_rel(a_psd_dict['p1'][:, freq_ <= 0.01].mean(axis = 1), 
          a_psd_dict['p2'][:, freq_ <= 0.01].mean(axis = 1))
# %%
