# %%
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img, get_data
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.surface import load_surf_mesh
import nibabel as nib
from neuromaps.datasets import fetch_fsaverage
import surfplot
import cmcrameri.cm as cmc
# %%
prjdir = "/work/O2Resting/"
rawdir = os.path.join(prjdir, "derivatives/fmriprep/")
outdir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")

# %%
# get mask
group_outdir = os.path.join(outdir, 'group')
group_mask = os.path.join(group_outdir, 'group_glmmask.nii.gz')
mask_data = get_data(group_mask)

# %%get atlas
arte_tpm = os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_ToF_Thresh.nii.gz')
arte_mask = resample_to_img(
    source_img= arte_tpm, 
    target_img = group_mask,
    interpolation = 'linear')
arte_data = get_data(arte_mask)

vein_tpm = os.path.join('../atlas/Template_MNI_2018_41_subjects/mean_Ved_swi_Thresh.nii.gz')
vein_mask = resample_to_img(
    source_img= vein_tpm, 
    target_img = group_mask,
    interpolation = 'linear')
vein_data = get_data(vein_mask)

# %%
p2p1_beta = os.path.join(group_outdir, 'p2p1', 'group_p2p1_beta.nii.gz')
p2p1_data = np.squeeze(get_data(p2p1_beta))

p3p1_beta = os.path.join(group_outdir, 'p3p1', 'group_p3p1_beta.nii.gz')
p3p1_data = np.squeeze(get_data(p3p1_beta))

p4p1_beta = os.path.join(group_outdir, 'p4p1', 'group_p4p1_beta.nii.gz')
p4p1_data = np.squeeze(get_data(p4p1_beta))

# %% get rois
n_rois = 100
schaefer_atlas = fetch_atlas_schaefer_2018(
    n_rois = n_rois, yeo_networks =  7, resolution_mm =1) # 1mm
schaefer_labels = [_.decode('utf-8') for _ in schaefer_atlas['labels']]
schaefer_network = np.array(
    [ re.match(r'^7Networks_(LH|RH)_(.+?)_(.+?)', _).group(2) 
     for _ in schaefer_labels ])
schaefer_dat = get_data(resample_to_img(
    source_img=schaefer_atlas['maps'],
    target_img=group_mask,
    interpolation='nearest'))
schaefer_mask = schaefer_dat.copy()
schaefer_mask[mask_data == 0] = 0
schaefer_mask[arte_data > 0.8] = 0 
schaefer_mask[vein_data > 0.8] = 0
# %%
for i in range(n_rois + 1):
    print(f'{i}: {np.sum(schaefer_mask == i)}')
# %%
p2p1_list = list()
p3p1_list = list()
p4p1_list = list()
for i in np.arange(1, n_rois + 1, 1):
    if np.sum(schaefer_mask == i) < 20:
        p2p1_list.append(np.nan)
    else:
        p2p1_list.append(np.median(p2p1_data[schaefer_mask == i]))
        
    if np.sum(schaefer_mask == i) < 20:
        p3p1_list.append(np.nan)
    else:
        p3p1_list.append(np.median(p3p1_data[schaefer_mask == i]))
        
    if np.sum(schaefer_mask == i) < 20:
        p4p1_list.append(np.nan)
    else:
        p4p1_list.append(np.median(p4p1_data[schaefer_mask == i]))
        
# %%
# preparation of plotting on surf
fs_lh_mesh = load_surf_mesh(os.path.abspath(os.path.join(
    "../atlas", "Schaefer_Parcellations/FreeSurfer5.3/fsaverage", 
    "surf/lh.inflated")))
fs_rh_mesh = load_surf_mesh(os.path.abspath(os.path.join(
    "../atlas", "Schaefer_Parcellations/FreeSurfer5.3/fsaverage", 
    "surf/rh.inflated")))

atlas_lh_label, _, atlas_lh_names   = nib.freesurfer.io.read_annot(os.path.abspath(os.path.join(
    "../atlas", "Schaefer_Parcellations/FreeSurfer5.3/fsaverage", 
    f"label/lh.Schaefer2018_{n_rois}Parcels_7Networks_order.annot")))
atlas_lh_names = np.array([str(_, "UTF-8") for _ in atlas_lh_names])

atlas_rh_label, _, atlas_rh_names   = nib.freesurfer.io.read_annot(os.path.abspath(os.path.join(
    "../atlas", "Schaefer_Parcellations/FreeSurfer5.3/fsaverage", 
    f"label/rh.Schaefer2018_{n_rois}Parcels_7Networks_order.annot")))
atlas_rh_names = np.array([str(_, "UTF-8") for _ in atlas_rh_names])
# %%
# plot bost p2-p1 on surface
cmap_max = np.max([ _ for _ in p2p1_list if ~np.isnan(_)])
cmap_min = np.min([ _ for _ in p2p1_list if ~np.isnan(_)])
grad_lh_data = np.zeros_like(atlas_lh_label).astype(float)
grad_rh_data = np.zeros_like(atlas_rh_label).astype(float)
for i_parc in range(len(schaefer_atlas['labels'])):
    parc_name = schaefer_atlas['labels'][i_parc].decode('utf-8')
    
    parc_idx = np.where(atlas_lh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_lh_data[atlas_lh_label == parc_idx
                     ] = p2p1_list[i_parc]
    parc_idx = np.where(atlas_rh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_rh_data[atlas_rh_label == parc_idx
                     ] = p2p1_list[i_parc]
        
surfaces = fetch_fsaverage(density="164k")
lh, rh = surfaces['inflated']
p = surfplot.Plot(lh, rh)
p.add_layer({'left': grad_lh_data, 'right': grad_rh_data},
            color_range = [cmap_min, cmap_max])
fig = p.build()
fig.axes[0].set_title(f"BOLD signal changes, p2-p1")
fig.show()
fig.savefig("./bold_diff_surf_p2p1.pdf")

# %%
# plot bost p3-p1 on surface
cmap_max = np.max(np.array(p3p1_list)*-1)
cmap_min = np.min(np.array(p3p1_list)*-1)
grad_lh_data = np.zeros_like(atlas_lh_label).astype(float)
grad_rh_data = np.zeros_like(atlas_rh_label).astype(float)
for i_parc in range(len(schaefer_atlas['labels'])):
    parc_name = schaefer_atlas['labels'][i_parc].decode('utf-8')
    
    parc_idx = np.where(atlas_lh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_lh_data[atlas_lh_label == parc_idx
                     ] = p3p1_list[i_parc] * -1
    parc_idx = np.where(atlas_rh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_rh_data[atlas_rh_label == parc_idx
                     ] = p3p1_list[i_parc] * -1
        
surfaces = fetch_fsaverage(density="164k")
lh, rh = surfaces['inflated']
p = surfplot.Plot(lh, rh)
p.add_layer({'left': grad_lh_data, 'right': grad_rh_data}, cmap = cmc.glasgow,
            color_range = [cmap_min, cmap_max])
fig = p.build()
fig.axes[0].set_title(f"BOLD signal changes, p1-p3")
fig.show()
fig.savefig("./bold_diff_surf_p3p1.pdf")
# %%
# plot bost p4-p1 on surface
cmap_max = np.max(p4p1_list)
cmap_min = np.min(p4p1_list)
grad_lh_data = np.zeros_like(atlas_lh_label).astype(float)
grad_rh_data = np.zeros_like(atlas_rh_label).astype(float)
for i_parc in range(len(schaefer_atlas['labels'])):
    parc_name = schaefer_atlas['labels'][i_parc].decode('utf-8')
    
    parc_idx = np.where(atlas_lh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_lh_data[atlas_lh_label == parc_idx
                     ] = p4p1_list[i_parc]
    parc_idx = np.where(atlas_rh_names == parc_name)[0]
    if len(parc_idx) > 0:
        grad_rh_data[atlas_rh_label == parc_idx
                     ] = p4p1_list[i_parc]
        
surfaces = fetch_fsaverage(density="164k")
lh, rh = surfaces['inflated']
p = surfplot.Plot(lh, rh)
p.add_layer({'left': grad_lh_data, 'right': grad_rh_data},
            color_range = [cmap_min, cmap_max])
fig = p.build()
fig.axes[0].set_title(f"BOLD signal changes, p4-p1")
fig.show()
fig.savefig("./bold_diff_surf_p4p1.pdf")

# %%
# prepare data as network
roibeta_info = dict({
    'label': [], 'network': []})

for i_roi in range(len(schaefer_labels)):
    # get roi, do dilation, turn to image
    roibeta_info['label'].append(schaefer_labels[i_roi])
    roibeta_info['network'].append(schaefer_network[i_roi])

roibeta_info['p2p1_psc'] = p2p1_list
roibeta_info['p3p1_psc'] = p3p1_list
roibeta_info['p4p1_psc'] = p4p1_list
tpm_pd = pd.DataFrame(roibeta_info)

# %%
network_order_p2p1 = tpm_pd.groupby(['network'])['p2p1_psc'].median().sort_values(ascending=False).index
network_order_p3p1 = tpm_pd.groupby(['network'])['p3p1_psc'].median().sort_values(ascending=False).index
network_order_p4p1 = tpm_pd.groupby(['network'])['p4p1_psc'].median().sort_values(ascending=False).index

# %%
# plot bost as netword
fig, axs = plt.subplots(1, 3, figsize = (12, 5))
sns.violinplot(
    data = tpm_pd, y = "p2p1_psc", x = 'network', ax = axs[0],
    order = network_order_p2p1, palette="Set2")
sns.violinplot(
    data = tpm_pd, y = "p3p1_psc", x = 'network', ax = axs[1],
    order = network_order_p2p1, palette="Set2")
sns.violinplot(
    data = tpm_pd, y = "p4p1_psc", x = 'network', ax = axs[2],
    order = network_order_p2p1, palette="Set2")
axs[0].set_xticklabels(network_order_p2p1, rotation = 90, ha = 'center', va = 'top')
axs[1].set_xticklabels(network_order_p2p1, rotation = 90, ha = 'center', va = 'top')
axs[2].set_xticklabels(network_order_p2p1, rotation = 90, ha = 'center', va = 'top')
plt.tight_layout()
fig.savefig("./bold_diff_network_p2p1_p3p1_p4p1.pdf")

# %% save the roi data for later network comparison
if not os.path.exists('./rois_betas_r'):
    os.makedirs('./rois_betas_r')
tpm_pd.to_csv('./rois_betas_r/rois_betas.tsv',
              sep = '\t', index = True)

# %%
# plot correlation between bost
fig, axs = plt.subplots(1, 3, figsize = (12, 4))
# axs 0
rhor, pval = pearsonr(tpm_pd["p2p1_psc"], tpm_pd["p3p1_psc"])
sns.scatterplot(tpm_pd, x = "p2p1_psc", y = "p3p1_psc", hue = "network",
                palette="Set2", ax = axs[0])
sns.regplot(tpm_pd, x = "p2p1_psc", y = "p3p1_psc", scatter = False,
            color = "black", ax = axs[0])
axs[0].set_title(f'r={rhor:.3f}; p={pval:.3f}')

#axs 1
rhor, pval = pearsonr(tpm_pd["p2p1_psc"], tpm_pd["p4p1_psc"])
sns.scatterplot(tpm_pd, x = "p2p1_psc", y = "p4p1_psc", hue = "network",
                palette="Set2", ax = axs[1])
sns.regplot(tpm_pd, x = "p2p1_psc", y = "p4p1_psc", scatter = False,
            color = "black", ax = axs[1])
axs[1].set_title(f'r={rhor:.3f}; p={pval:.3f}')

#axs 2
rhor, pval = pearsonr(tpm_pd["p3p1_psc"], tpm_pd["p4p1_psc"])
sns.scatterplot(tpm_pd, x = "p3p1_psc", y = "p4p1_psc", hue = "network",
                palette="Set2", ax = axs[2])
sns.regplot(tpm_pd, x = "p3p1_psc", y = "p4p1_psc", scatter = False,
            color = "black", ax = axs[2])
axs[2].set_title(f'r={rhor:.3f}; p={pval:.3f}')
plt.tight_layout()
fig.savefig("./bold_diff_cor_p2p1_p3p1_p4p1.pdf")
# %%
