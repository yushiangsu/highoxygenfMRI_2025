# %%
import os
import re
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import plot_matrix
import matplotlib.pyplot as plt

# %%
prjdir = "/work/O2Resting/"
outdir = os.path.join(prjdir, "derivatives/nilearn_funcconn/")

# %% prepare Schaefer atlas/parcellations
atlas_schaefer = fetch_atlas_schaefer_2018(n_rois = 100, yeo_networks = 7)
schaefer_labels = [
    _.decode('utf-8') for _ in atlas_schaefer['labels'] ]
schaefer_networks = [
    re.match(r'^7Networks_(LH|RH)_(.+?)_(.+?)', _).group(2) 
    for _ in schaefer_labels ]

# %%
roi_idx = list()
net_labels = np.unique(schaefer_networks)
for i_net in net_labels:
    roi_idx.append(
        [ i_roi for i_roi in range(len(schaefer_networks)) 
         if schaefer_networks[i_roi] == i_net ])
roi_reidx = np.concatenate(roi_idx)

# %%
group_outdir = os.path.join(outdir, 'group')
p2p1_con_mask = np.loadtxt(
    os.path.join(group_outdir, 'p2p1_conmat.txt'), delimiter=",")
p2p1_stats = np.loadtxt(
    os.path.join(group_outdir, 'p2p1_constat.txt'), delimiter=",")
p2p1_stats[p2p1_con_mask == 0] = 0

p3p1_con_mask = np.loadtxt(
    os.path.join(group_outdir, 'p3p1_conmat.txt'), delimiter=",")
p3p1_stats = np.loadtxt(
    os.path.join(group_outdir, 'p3p1_constat.txt'), delimiter=",")
p3p1_stats[p3p1_con_mask == 0] = 0

p4p1_con_mask = np.loadtxt(
    os.path.join(group_outdir, 'p4p1_conmat.txt'), delimiter=",")
p4p1_stats = np.loadtxt(
    os.path.join(group_outdir, 'p4p1_constat.txt'), delimiter=",")
p4p1_stats[p4p1_con_mask == 0] = 0

# %%
# No significant difference for p3p1
# Just plot p2p1 and p4p1

cmax = np.array([np.max(p2p1_stats[p2p1_stats != 0.0]),
                 np.max(p4p1_stats[p4p1_stats != 0.0])]).max()
cmin = np.array([np.min(p2p1_stats[p2p1_stats != 0.0]),
                 np.min(p4p1_stats[p4p1_stats != 0.0])]).min()
p2p1_stats_mask = p2p1_stats.copy()
p2p1_stats_mask[p2p1_stats_mask > 0] = np.nan
p4p1_stats_mask = p4p1_stats.copy()
p4p1_stats_mask[p4p1_stats_mask > 0] = np.nan
fig, axs = plt.subplots(1, 2, figsize = (10, 6))
plot_matrix(
    p2p1_stats[roi_reidx, :][:,roi_reidx], title = "p2 - p1",
    vmax = cmax, vmin = 0, axes = axs[0], cmap = plt.cm.YlOrRd,
    reorder = False)
plot_matrix(
    p2p1_stats_mask[roi_reidx, :][:,roi_reidx], title = "p2 - p1",
    vmax = cmax, vmin = -cmax, axes = axs[0], colorbar = False,
    reorder = False)
plot_matrix(
    p4p1_stats[roi_reidx, :][:,roi_reidx], title = "p4 - p1",
    vmax = cmax, vmin = 0, axes = axs[1], cmap = plt.cm.YlOrRd,
    reorder = False)
plot_matrix(
    p4p1_stats_mask[roi_reidx, :][:,roi_reidx], title = "p4 - p1",
    vmax = cmax, vmin = -cmax, axes = axs[1], colorbar = False,
    reorder = False)

for _line in np.cumsum([len(_) for _ in roi_idx]):
    axs[0].axvline(x = _line - 0.5, color = 'black')
    axs[1].axvline(x = _line - 0.5, color = 'black')
    axs[0].axhline(y = _line - 0.5, color = 'black')
    axs[1].axhline(y = _line - 0.5, color = 'black')
    
axs[0].set_xticks(np.cumsum([len(_) for _ in roi_idx]) - np.array([len(_) for _ in roi_idx])/2)
axs[0].set_xticklabels(net_labels, rotation = 90)
axs[1].set_xticks(np.cumsum([len(_) for _ in roi_idx]) - np.array([len(_) for _ in roi_idx])/2)
axs[1].set_xticklabels(net_labels, rotation = 90)
axs[0].set_yticks(np.cumsum([len(_) for _ in roi_idx]) - np.array([len(_) for _ in roi_idx])/2)
axs[0].set_yticklabels(net_labels)
plt.tight_layout()
plt.savefig("./funcconn_p2p1_p4p1.pdf")