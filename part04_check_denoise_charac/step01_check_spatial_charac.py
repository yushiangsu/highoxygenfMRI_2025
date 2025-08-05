# %%
import os
import re
import pandas as pd
import numpy as np
from nilearn.image import get_data, resample_to_img, math_img
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "derivatives/fmriprep/")
outdir = os.path.join(prjdir, "derivatives/melodic_ica/")
tpmdir = os.path.join(prjdir, "derivatives/ants_tpmatsub/")

# %%
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")

# %%
all_icomp_ratio = list()
for i_sub in subinfo['participant_id']:
    print(f'Working on {i_sub}...')
    for i_run in [1, 2]:
        #
        # load boldref as reference
        boldmask_path = os.path.join(
            datdir, f'{i_sub}', 'func', f'{i_sub}_task-rest_run-{i_run}_space-T1w_desc-brain_mask.nii.gz')
        boldmask_dat = get_data(boldmask_path)
        tiss_info = {
            'gm': {
                'subimg': os.path.join(
                    tpmdir, 'gm', f'{i_sub}', 
                    f'{i_sub}_space-T1w_label-GM_probseg.nii.gz')
            },
            'wm': {
                'subimg': os.path.join(
                    tpmdir, 'wm', f'{i_sub}', 
                    f'{i_sub}_space-T1w_label-WM_probseg.nii.gz')
            },
            'csf': {
                'subimg': os.path.join(
                    tpmdir, 'csf', f'{i_sub}', 
                    f'{i_sub}_space-T1w_label-CSF_probseg.nii.gz')
            },
            'arte': {
                'subimg': os.path.join(
                        tpmdir, 'arte', f'{i_sub}', 
                        f'{i_sub}_space-T1w_ToF_Thresh.nii.gz')
            },
            'vein': {
                'subimg': os.path.join(
                        tpmdir, 'vein', f'{i_sub}', 
                        f'{i_sub}_space-T1w_swi_Thresh.nii.gz')
            },
        }
        tiss_tpmprob = dict()
        for tiss_name, tiss_dict in tiss_info.items():
            tiss_tpm_resample = resample_to_img(
                source_img = tiss_dict['subimg'],
                target_img = boldmask_path,
                interpolation = 'continuous'
            )

            tiss_tpmprob[tiss_name] = get_data(tiss_tpm_resample)
        
        # get melodic output
        icomp_path = os.path.join(
            outdir, f'{i_sub}', f'space-T1w_run-{i_run}.ica', 
            'filtered_func_data.ica', 'melodic_IC.nii.gz')
        icomp_dat = get_data(icomp_path)

        #
        icomp_fixout = list()
        with open(
            os.path.join(
                outdir, f'{i_sub}', f'space-T1w_run-{i_run}.ica', 
                'fix4melview_Standard_thr20.txt'), 
            'r') as f:
            for txt_line in f.readlines():
                txt_re = re.match('^([0-9]+), (.+), (.+)\n', txt_line)
                if txt_re:
                    icomp_fixout.append(
                        [txt_re.group(1), txt_re.group(2), txt_re.group(3)])
        icomp_fixout = pd.DataFrame(
            np.array(icomp_fixout), columns = ['idx', 'label', 'regout'])
        #
        icomp_ratio = dict({
            'arte': [], 'vein': [], 'gm': [], 'wm': [], 'csf': [],
            'fix': [],
        })
        for i_icomp in range(icomp_dat.shape[3]):
            pick_icomp = icomp_dat[:, :, :, i_icomp]
            nvox_icomp = np.sum((pick_icomp > 3) | (pick_icomp < -3))
            for tiss_name in tiss_tpmprob.keys():
                icomp_ratio[tiss_name].append(
                    np.mean(tiss_tpmprob[tiss_name][(pick_icomp > 3) | (pick_icomp < -3)]))

            icomp_ratio['fix'].append(icomp_fixout['regout'][i_icomp])
        
        icomp_ratio = pd.DataFrame(icomp_ratio)
        
        ## standardized these value within one run for later clustering method
        #icomp_ratio['arteX'] = (icomp_ratio['arte'] - np.mean(icomp_ratio['arte'])
        #                        )/np.std(icomp_ratio['arte'])
        #icomp_ratio['veinX'] = (icomp_ratio['vein'] - np.mean(icomp_ratio['vein'])
        #                        )/np.std(icomp_ratio['vein'])
        #icomp_ratio['gmX'] = (icomp_ratio['gm'] - np.mean(icomp_ratio['gm'])
        #                        )/np.std(icomp_ratio['gm'])
        #icomp_ratio['wmX'] = (icomp_ratio['wm'] - np.mean(icomp_ratio['wm'])
        #                        )/np.std(icomp_ratio['wm'])
        #icomp_ratio['csfX'] = (icomp_ratio['csf'] - np.mean(icomp_ratio['csf'])
        #                        )/np.std(icomp_ratio['csf'])
        
        icomp_ratio['sub'] = i_sub
        icomp_ratio['run'] = i_run
        icomp_ratio['idx'] = np.arange(icomp_dat.shape[3])
        
        # append to big data
        all_icomp_ratio.append(icomp_ratio.copy())

# aggregate all data
all_icomp_ratio = pd.concat(all_icomp_ratio, axis = 0).reset_index(drop = True)

# %%
# save data for glmm testing
if not os.path.exists('./icomp_glmm_r'):
    os.makedirs('./icomp_glmm_r')
all_icomp_ratio.to_csv('./icomp_glmm_r/icomp_ratio.tsv', 
                       sep = '\t', index = False)

# %%
# plot percentage of removed components for each participant
plt1_dat = all_icomp_ratio.groupby(['sub', 'run']).apply(
    lambda df: np.sum(df['fix'] == 'True') / df.shape[0]
).reset_index(name = 'prob_of_noise')

fig, axs = plt.subplots(1, figsize = (3, 5))
sns.violinplot(plt1_dat, y = 'prob_of_noise', ax = axs)
sns.stripplot(plt1_dat, y = 'prob_of_noise', size = 10, jitter = 0.3, 
              color = '.2', alpha = 0.5, ax = axs)
axs.set_ylim(0.15, 0.75)
axs.set_ylabel('Proportion of Removed Components')
plt.tight_layout()
fig.savefig("ica_spatial_plot1.pdf")
# %%
plt2_dat = all_icomp_ratio[['vein', 'arte', 'gm', 'wm', 'csf', 'fix']].rename(
    columns={'vein': 'Veins',
             'arte': 'Arteries',
             'gm': 'GM',
             'wm': 'WM',
             'csf': 'CSF'})
plt2_dat['id'] = plt2_dat.index
plt2_dat = pd.melt(plt2_dat, id_vars = ['id', 'fix'], 
                   value_vars = ['Veins', 'Arteries', 'GM', 'WM', 'CSF'],
                   var_name = 'structure', value_name = 'prob')
plt2_dat['prob'] = plt2_dat['prob'] * 100

struc_list = ['Veins', 'Arteries', 'GM', 'WM', 'CSF']
fig, axs = plt.subplots(1, 5, figsize = (12, 4.8))
for i_struc, struc_name in enumerate(struc_list):
    sns.barplot(
        plt2_dat.loc[plt2_dat["structure"] == f'{struc_name}'], 
        x = 'fix', y = 'prob',
        errorbar = 'se', ax = axs[i_struc])
    axs[i_struc].set_ylabel('')
    axs[i_struc].set_title(f'{struc_name}')

axs[0].set_ylabel('Proportion of voxels (%)')
plt.tight_layout()
fig.savefig("ica_spatial_plot2.pdf")

## %%
## statistic for gm
#ttest_ind(
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'True']['gm'].values,
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'False']['gm'].values)
## %%
## statistic for wm
#ttest_ind(
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'True']['wm'].values,
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'False']['wm'].values)
## %%
## statistic for csf
#ttest_ind(
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'True']['csf'].values,
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'False']['csf'].values)
## %%
## statistic for vein
#ttest_ind(
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'True']['vein'].values,
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'False']['vein'].values)
## %%
## statistic for arte
#ttest_ind(
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'True']['arte'].values,
#    all_icomp_ratio.loc[all_icomp_ratio['fix'] == 'False']['arte'].values)

# %%

## %%
## hierarchy clustering on the components given by the topographic 
## probabilities of each tissue. We identify 5 clusters. Specifically, 
## one cluster have high arte, vein along. And, another cluster with high 
## csf accompanied with high arte, vein. We believe these two clusters had 
## the componenets specific associated with vascular artifacts. FIX identified
## 75% and 81% of these two clusters as noise artifact. It suggest FIX did
## identified noise components associated with vascular. 
#from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster import hierarchy
## %%
#def plot_dendrogram(model, **kwargs):
#    # create the counts of samples under each node
#    counts = np.zeros(model.children_.shape[0])
#    n_samples = len(model.labels_)
#    for i, merge in enumerate(model.children_):
#        current_count = 0
#        for child_idx in merge:
#            if child_idx < n_samples:
#                current_count += 1  # leaf node
#            else:
#                current_count += counts[child_idx - n_samples]
#        counts[i] = current_count
#
#    linkage_matrix = np.column_stack(
#        [model.children_, model.distances_, counts]
#    ).astype(float)
#
#    # Plot the corresponding dendrogram
#    hierarchy.dendrogram(linkage_matrix, **kwargs)
## %%
#inputX = all_icomp_ratio.loc[:, ['arteX', 'veinX', 'gmX', 'wmX', 'csfX']].values
#hieClus = AgglomerativeClustering(
#    distance_threshold=0, n_clusters=None).fit(inputX)
## %%
#fig, axs = plt.subplots(1, figsize = (15, 8))
#plot_dendrogram(hieClus, ax = axs)
## %%
#for n_clus in [3, 4, 5, 6, 7, 8]:
#    fixratio = list()
#    clustout = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clus).fit_predict(inputX)
#    for i_clus in range(n_clus):
#        fixratio.append(
#            np.sum((clustout == i_clus) & (all_icomp_ratio['fix'] == 'True')) / 
#            np.sum(clustout == i_clus))
#        print(inputX[clustout == i_clus, :].mean(axis = 0))
#    print(fixratio)
## %%
## pick cluster = 5 
##
#n_clusters = 5
#clustout = AgglomerativeClustering(
#    distance_threshold=None, n_clusters=n_clusters).fit_predict(inputX)
#for i_clus in range(n_clusters):
#    fixratio.append(
#        np.sum((clustout == i_clus) & (all_icomp_ratio['fix'] == 'True')) / 
#        np.sum(clustout == i_clus))
#    print(inputX[clustout == i_clus, :].mean(axis = 0))
## %%
#fig, axs = plt.subplots(1, n_clusters + 1, figsize = (15, 5))
#fixratio = list()
#for i_clus in range(n_clusters):
#    fixratio.append(
#        np.sum((clustout == i_clus) & (all_icomp_ratio['fix'] == 'True')) / 
#        np.sum(clustout == i_clus))
#    axs[i_clus].bar(
#        x = np.arange(5), 
#        height = all_icomp_ratio.loc[clustout == i_clus][[
#            'gm', 'wm', 'csf', 'arte', 'vein']].mean(),
#        color = ['#a559aa', '#59a89c', '#f0c571', '#e02b35', '#082a54']
#        )
#    axs[i_clus].errorbar(
#        x = np.arange(5), 
#        y = all_icomp_ratio.loc[clustout == i_clus][[
#            'gm', 'wm', 'csf', 'arte', 'vein']].mean(),
#        yerr = all_icomp_ratio.loc[clustout == i_clus][[
#        'gm', 'wm', 'csf', 'arte', 'vein']].std(),
#        elinewidth = 2, linewidth = 0,
#        ecolor = ['#a559aa', '#59a89c', '#f0c571', '#e02b35', '#082a54']
#    )
#    axs[i_clus].set_title(f'Cluster {i_clus + 1}')
#    axs[i_clus].set_xlabel(f'Tissue')
#    axs[i_clus].set_xticks(np.arange(5))
#    axs[i_clus].set_xticklabels(['gm', 'wm', 'csf', 'arte', 'vein'])
#    axs[i_clus].set_ylabel(f'Mean Probability (Occupancy) +- SD')
#    axs[i_clus].set_ylim((0.00, 0.52))
#axs[n_clusters].bar(
#    x = np.arange(n_clusters),
#    height = fixratio,
#    color = 'black'
#)
#axs[n_clusters].set_title(f'Noise Components \nidentified by FIX')
#axs[n_clusters].set_xlabel(f'Cluster')
#axs[n_clusters].set_xticks(np.arange(n_clusters))
#axs[n_clusters].set_xticklabels([ _+1 for _ in range(n_clusters)])
#axs[n_clusters].set_ylabel(f'percentage of compoenents')
#axs[n_clusters].set_yticks(np.arange(0, 0.85, 0.1))
#axs[n_clusters].set_yticklabels([f'{int(_*100):d}%' for _ in np.arange(0, 0.85, 0.1)])
#plt.tight_layout()