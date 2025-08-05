# %%
import os
import re
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, pearsonr

from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.decode.discrete import ROIAssociationDecoder
from nimare.decode.continuous import CorrelationDecoder
from nimare.meta.cbma import mkda

from nilearn.image import get_data, new_img_like, resample_to_img, math_img
from nilearn.datasets import fetch_atlas_schaefer_2018

import matplotlib.pyplot as plt
import seaborn as sns

# %% prepare data path
prjdir = "/work/O2Resting/"
glmdir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")
db_dir = os.path.join("./neurosynth_db")

if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# %%
target_cont = 'p3p1'

# %% get neurosynth data set
nslda50_db = fetch_neurosynth(
    data_dir = db_dir,
    version = "7",
    overwrite = False,
    source = "abstract",
    vocab = "LDA50"
)
nslda50_dset = convert_neurosynth_to_dataset(
    coordinates_file=nslda50_db[0]["coordinates"],
    metadata_file=nslda50_db[0]["metadata"],
    annotations_files=nslda50_db[0]["features"],
)
nslda50_labels = np.array([
    re.sub(r'LDA50_abstract_weight__', 'Topic', _) 
    for _ in nslda50_dset.get_labels()])

nslda50_newlab = np.array([
    'Topic00, (Resting-State Default Mode Network)',
    'Topic01, (Personality and Anxiety Traits)', 
    'Topic02, (Cerebellar and Basal Ganglia Function)', 
    'Topic03, (Cingulate Cortex and Insular)', 
    'Topic04, (Stimulus Processing and Temporal Dynamics)', 
    'Topic05, (Frontal and Parietal Cortical Networks)', 
    'Topic06, Auditory Processing and Speech Perception', 
    'Topic07, Reward Processing and Motivation', 
    'Topic08, Social Reasoning and Theory of Mind', 
    'Topic09, Working Memory and Cognitive Load', 
    'Topic10, (Food and Eating Behaviors)', 
    'Topic11, (Learning, Training, and Skill Acquisition)', 
    'Topic12, (Gender Differences and Sexual Dimorphism)', 
    'Topic13, Fear, Threat, and Addiction', 
    'Topic14, (Neurodegenerative Diseases)', 
    'Topic15, (Cognitive Performance and Task Demands)', 
    'Topic16, Response Inhibition and Motor Control', 
    'Topic17, Motor Imagery and Execution', 
    'Topic18, Numerical Cognition and Arithmetic Processing', 
    'Topic19, Action Observation and Mirror Neurons', 
    'Topic20, Cognitive Control and Conflict Resolution', 
    'Topic21, (Structural Brain Imaging and Morphometry)', 
    'Topic22, (Neuroimaging Methods and Variability)', 
    'Topic23, (Autism Spectrum Disorder and Social Cognition)', 
    'Topic24, (Age and Developmental Differences)', 
    'Topic25, Spatial Representation and Body Mapping', 
    'Topic26, Emotional Processing and Regulation', 
    'Topic27, (Neurodevelopmental and Psychiatric Disorders)', 
    'Topic28, Social Cognition and Empathy', 
    'Topic29, (Stress, Trauma, and PTSD)', 
    'Topic30, Decision-Making and Risk Evaluation', 
    'Topic31, [Predictive Modeling and Classification]', 
    'Topic32, Pain Perception and Somatosensory Processing', 
    'Topic33, Memory Encoding and Retrieval', 
    'Topic34, (Neuronal Oscillations and Temporal Dynamics)', 
    'Topic35, (Genetic Risk Factors in Psychiatric disorder)', 
    'Topic36, (Pharmacological Treatments and Neurotransmitter Response)', 
    'Topic37, Language and Reading Processing', 
    'Topic38, Semantic and Conceptual Representations', 
    'Topic39, (Neurostimulation and Modulation Techniques)', 
    'Topic40, Facial Expression and Emotion Recognition', 
    'Topic41, Mental Imagery and Spatial Navigation', 
    'Topic42, Visual Perception and Multisensory Integration', 
    'Topic43, (Magnetic Resonance Imaging Mechanisms and Models)', 
    'Topic44, (Eye Movements and Sleep)', 
    'Topic45, Motion Perception and Visual Illusions', 
    'Topic46, (Hemispheric Lateralization and Language Recovery)', 
    'Topic47, Attention', 
    'Topic48, Prefrontal Cortex and Executive Functions', 
    'Topic49, (Depression and Resting-State Brain Activity)', 
])
# %% load univariate glm results
result_map = os.path.join(glmdir, 'group', f'{target_cont}', f'group_{target_cont}_beta.nii.gz')
result_mask = os.path.join(glmdir, 'group', 'group_glmmask.nii.gz')

# reverse the sign because p3-p1 is suppression
if target_cont == 'p3-p1':
    result_dat = get_data(result_map) * -1
else:
    result_dat = get_data(result_map)
mask_dat = get_data(result_mask)
# %% get atlas
gmmask_img = math_img(
    "(gm_img > 0.8) & (brain_mask > 0.5)",
    gm_img = resample_to_img(
        source_img = os.path.join('../atlas/MNI_TPM/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz'),
        target_img = result_mask,
        interpolation = 'continuous'),
    brain_mask = result_mask)
gmmask_dat = get_data(gmmask_img)

# prepare new maps
result_dat[mask_dat < 0.5] = 0.0
result_newmap = new_img_like(ref_niimg = result_map, data = result_dat)

## %% Method 1: create schaefer's 100 ROIs
#n_rois = 100
#n_bins = 20
#n_study = nslda50_dset.annotations.shape[0]
#
#schaefer_atlas = fetch_atlas_schaefer_2018(
#    n_rois = n_rois, yeo_networks =  7, resolution_mm = 1) # 1mm
#schaefer_labels = np.array([ _.decode('utf-8') for _ in schaefer_atlas['labels']])
#schaefer_dat = get_data(resample_to_img(
#    source_img = schaefer_atlas['maps'],
#    target_img = result_mask, 
#    interpolation = "nearest"))
#schaefer_dat[mask_dat < 1] = 0
#
#schaefer_values = np.array(
#    [result_dat[schaefer_dat == (_)].mean() for _ in range(n_rois + 1)]
#    ) # the first value is background
#bins_perc = np.percentile(schaefer_values[1:], q = np.linspace(0, 100, n_bins+1))
#
#decoded_dfs = list()
#for i_bin in range(n_bins):
#    if i_bin == 0:
#        bin_idx = np.where(
#            (schaefer_values <= bins_perc[i_bin+1])
#            )[0]
#    else:
#        bin_idx = np.where(
#            (schaefer_values <= bins_perc[i_bin+1]) &
#            (schaefer_values > bins_perc[i_bin])
#            )[0]
#    bin_img_dat = np.stack(
#        [schaefer_dat == (_) for _ in bin_idx if _ != 0], axis = 3).sum(axis = 3)
#    bin_img = new_img_like(result_mask, bin_img_dat)
#    
#    # decode
#    metadecoder = ROIAssociationDecoder(bin_img)
#    metadecoder.fit(nslda50_dset)
#    decoded_df = metadecoder.transform()
#    decoded_dfs.append(decoded_df.copy())
#
## %%
#decoded_all = pd.concat(decoded_dfs, axis = 1)
#decoded_zval = np.arctanh(decoded_all.values)
#decoded_smo = gaussian_filter1d(decoded_zval, axis = 1, sigma= 0.5)
#decoded_sortidx = np.argsort(np.argmax(decoded_smo, axis = 1))
#zval_th = norm.ppf(0.99, loc = 0, scale = 1/np.sqrt(n_study - 3))
#decoded_zval[decoded_zval < zval_th] = 0
#invalid_idx = np.sum(decoded_zval > 0, axis = 1) < n_bins/5 
#invalid_idx = np.logical_or(
#    invalid_idx, 
#    np.array([ True if _.endswith(')') else False for _ in nslda50_newlab ]))
#decoded_trim = decoded_zval[decoded_sortidx, :][~invalid_idx[decoded_sortidx], :]
#decoded_label = nslda50_newlab[decoded_sortidx][~invalid_idx[decoded_sortidx]]
## %%
#plt_dat = pd.DataFrame(
#    decoded_trim, index = [ _[9:] for _ in decoded_label], 
#    columns = [f'Bin_{_+1:02d}' for _ in range(n_bins)])
#fig, axs = plt.subplots(1, figsize = (10, 10))
#axs = sns.heatmap(plt_dat, vmin = 0, vmax = 0.25)
#axs.set_xticks(range(n_bins + 1))
#axs.set_xticklabels([f'{int(_):3d}%' for _ in np.arange(0, 100+1, 100/n_bins)])
#axs.tick_params(axis='y', labelsize=16)
#axs.tick_params(axis='x', labelsize=16)
# %% method 2: continous decoding
metacontdecoder = CorrelationDecoder(
    frequency_threshold=0.005,
    meta_estimator=mkda.MKDAChi2,
    target_image='z_desc-association'
)
metacontdecoder.fit(nslda50_dset)
decoded_rval = metacontdecoder.transform(result_newmap)
result_zval = metacontdecoder.results_.masker.transform(result_newmap).squeeze()

decoded_rdict = dict()
for dec_k, dec_v in metacontdecoder.results_.maps.items():
    topic_re = re.search(r'LDA50_abstract_weight__([0-9]+)_.*', dec_k)
    topic_idx = int(topic_re.group(1))
    decoded_rdict[dec_k] = dict()
    rcorr = pearsonr(dec_v, result_zval)
    conflw, confup = rcorr.confidence_interval(0.95)
    decoded_rdict[dec_k]['rhor'] = rcorr.statistic
    decoded_rdict[dec_k]['pval'] = rcorr.pvalue
    decoded_rdict[dec_k]['newlab'] = nslda50_newlab[topic_idx]
    decoded_rdict[dec_k]['conflw'] = conflw
    decoded_rdict[dec_k]['confup'] = confup

with open(f"./{target_cont}_corrdecoder.json", mode = "wt", encoding = "utf-8") as f:
    json.dump(decoded_rdict, f, ensure_ascii = False, indent = 2)
    
# %%
plt_dat = dict({
    'label': [],
    'rhor': [],
    'conflw': [],
    'confup': []
})
for dec_k, dec_v in decoded_rdict.items():
    if (dec_v['rhor'] > 0) & (dec_v['pval'] < (0.05/len(decoded_rdict.keys()))):
        #plt_dat['label'].append(re.sub(r'LDA50_abstract_weight__', 'Topic', dec_k))
        plt_dat['label'].append(dec_v['newlab'])
        plt_dat['rhor'].append(dec_v['rhor'])
        plt_dat['conflw'].append(dec_v['conflw'])
        plt_dat['confup'].append(dec_v['confup'])
plt_dat = pd.DataFrame(plt_dat).sort_values('rhor', ascending=False).reset_index(drop = True)

# %%
fig, axs = plt.subplots(1, figsize = (10, 15))
axs = sns.barplot(plt_dat, y = "label", x = "rhor")
axs.errorbar(x = plt_dat['rhor'],
             y = plt_dat['label'],
             xerr = np.array([
                 list(plt_dat['rhor'] - plt_dat['conflw']), 
                 list(plt_dat['confup'] - plt_dat['rhor'])]),
             fmt = 'none',
             ecolor = 'black')
axs.set_xlim(0, 0.295)
axs.set_xticklabels([0, 0.05, 0.10, 0.15, 0.20, 0.25])
plt.tight_layout()
fig.savefig(f"funcdeco_{target_cont}.pdf")
# %%
