# %%
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img, index_img, math_img, mean_img, resample_to_img
from nilearn.masking import compute_epi_mask
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import view_img

# %% prepare data path
prjdir = "/work/O2Resting/"
rawdir = os.path.join(prjdir, "derivatives/fmriprep/")
datdir = os.path.join(prjdir, "derivatives/fix_agg_originalcomps/")
outdir = os.path.join(prjdir, "derivatives/nilearn_univglm_agg_originalcomps_psc/")
if not os.path.exists(outdir):
    os.makedirs(outdir)

subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")

# %% helpful function to create design matrix
def get_dm(i_sub, i_run):
    total_tr = 600
    tr = 2

    # sub-010 run-1 delayed 10 second to switch
    if (i_sub == '010') and (i_run == 1):
        p1_tr = 160
        p2_tr = 140
        p3_tr = 150
    else:
        p1_tr = 150
        p2_tr = 150
        p3_tr = 150

    event_dm = pd.DataFrame({
        'p1': np.concatenate([
            np.ones(p1_tr),
            np.zeros(p2_tr),
            np.zeros(p3_tr),
            np.zeros(total_tr - p1_tr - p2_tr - p3_tr)]),
        'p2': np.concatenate([
            np.zeros(p1_tr),
            np.ones(p2_tr),
            np.zeros(p3_tr),
            np.zeros(total_tr - p1_tr - p2_tr - p3_tr)]),
        'p3': np.concatenate([
            np.zeros(p1_tr),
            np.zeros(p2_tr),
            np.ones(p3_tr),
            np.zeros(total_tr - p1_tr - p2_tr - p3_tr)]),
        'p4': np.concatenate([
            np.zeros(p1_tr),
            np.zeros(p2_tr),
            np.zeros(p3_tr),
            np.ones(total_tr - p1_tr - p2_tr - p3_tr)])
        }, index = np.arange((0 + 0.5)*tr, (total_tr + 0.5)*tr, step = tr))
        
    return event_dm
# %%
# calculate baseline BOLD signal for the first five minutes = 150 TR
base_idx = slice(0, 150)
# %%
cmap_list = dict({
    'p2p1': [],
    'p3p1': [],
    'p4p1': [],
    'mask': [],
    'glmmask': [],
})
for i_sub in subinfo['participant_id']:
    subid = i_sub
    #subid = 'sub-001'
    print(f'Working on {subid}!')
    sub_rawdir = os.path.join(rawdir, f'{subid}')
    sub_datdir = os.path.join(datdir, f'{subid}')
    sub_outdir = os.path.join(outdir, f'{subid}')
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)
    
    sub_inputs = dict({
        'bolds': [],
        'masks': [],
        'evets': [],
        'glmmasks': [],
    })
    for i_run in [1, 2]:
        # Load data
        raw_path = os.path.join(
            sub_rawdir, 'func',
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        bold_img = os.path.join(
            sub_datdir,
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-denoise_bold.nii.gz')
        mask_img = math_img(
            '(mask_img > 0.5)',
            mask_img = os.path.join(
                sub_datdir,
                f'{subid}_task-rest_run-{i_run}_'
                f'space-MNI152NLin2009cAsym_desc-denoise_mask.nii.gz'))
        rawepi_mask = compute_epi_mask(
            os.path.join(
                rawdir, f'{subid}', 'func',
                f'{subid}_task-rest_run-{i_run}_'
                f'space-MNI152NLin2009cAsym_boldref.nii.gz'),
            lower_cutoff = 0.2, upper_cutoff = 0.5)
        brain_mask = resample_to_img(
            source_img = load_mni152_brain_mask(resolution=1),
            target_img = rawepi_mask, interpolation='nearest')
        glmmask_img =  math_img('(mask_img > 0.5) & (brain_mask > 0.5)',
            mask_img = rawepi_mask, brain_mask = brain_mask)
        sub_inputs['masks'].append(mask_img)
        sub_inputs['glmmasks'].append(glmmask_img)
        
        # calculate Percent signal change
        meanbase = math_img(f'np.mean(imgs, axis = 3, keepdims = True)'
                            f'* (mask_img[:, :, :, np.newaxis] > 0.5)',
                            imgs = index_img(bold_img, base_idx),
                            mask_img = mask_img)
        psc_img = math_img(
            f'(bold_img - meanbase) * '
            f'np.divide(1, meanbase, out=np.zeros_like(meanbase), where=(meanbase!=0))'
            f'* 100 * (mask_img[:, :, :, np.newaxis] > 0.5)',
            bold_img = bold_img, mask_img = mask_img,
            meanbase = meanbase)
        psc_img.to_filename(os.path.join(
            sub_datdir,
            f'{subid}_task-rest_run-{i_run}_'
            f'space-MNI152NLin2009cAsym_desc-denoise_boldpsc.nii.gz'))
        sub_inputs['bolds'].append(load_img(psc_img))
        
        # Load events
        evet_pd = get_dm(subid, i_run)
        sub_inputs['evets'].append(evet_pd)

    # get mask
    mask_img = math_img('meanimg > 0.5', meanimg = mean_img(sub_inputs['masks']))
    cmap_list['mask'].append(mask_img)
    glmmask_img = math_img('meanimg > 0.5', meanimg = mean_img(sub_inputs['glmmasks']))
    cmap_list['glmmask'].append(glmmask_img)
    
    # Using first-level model to clean out the effect of motion
    diffglm = FirstLevelModel(
        mask_img = mask_img,
        noise_model = 'ar1',
        signal_scaling = False,
        minimize_memory = False)

    diffglm.fit(sub_inputs['bolds'], 
                design_matrices = sub_inputs['evets'])
    
    p2p1_diffbeta = diffglm.compute_contrast('p2-p1', output_type = 'effect_size')
    p2p1_diffbeta.to_filename(os.path.join(
        sub_outdir,
        f'{subid}_space-MNI152NLin2009cAsym_desc-p2p1diff_beta.nii.gz'))
    cmap_list['p2p1'].append(p2p1_diffbeta)

    p3p1_diffbeta = diffglm.compute_contrast('p3-p1', output_type = 'effect_size')
    p3p1_diffbeta.to_filename(os.path.join(
        sub_outdir,
        f'{subid}_space-MNI152NLin2009cAsym_desc-p3p1diff_beta.nii.gz'))
    cmap_list['p3p1'].append(p3p1_diffbeta)
    
    p4p1_diffbeta = diffglm.compute_contrast('p4-p1', output_type = 'effect_size')
    p4p1_diffbeta.to_filename(os.path.join(
        sub_outdir,
        f'{subid}_space-MNI152NLin2009cAsym_desc-p4p1diff_beta.nii.gz'))
    cmap_list['p4p1'].append(p4p1_diffbeta)
        
# %%
n_sub = subinfo.shape[0]
group_mask = math_img(f'img > (1 - 1/{n_sub})', 
                      img = mean_img(cmap_list['mask']))
group_glmmask = math_img(f'img > (1 - 1/{n_sub})', 
                      img = mean_img(cmap_list['glmmask']))

# %%
group_outdir = os.path.join(outdir, 'group_mask')
if not os.path.exists(group_outdir):
    os.makedirs(group_outdir)
    
group_mask.to_filename(os.path.join(group_outdir, 'group_mask.nii.gz'))
group_glmmask.to_filename(os.path.join(group_outdir, 'group_glmmask.nii.gz'))
# %%
# group statistic on P2-P1
if not os.path.exists(os.path.join(group_outdir, 'p2p1')):
    os.makedirs(os.path.join(group_outdir, 'p2p1'))
secondlvlglm_p2p1 = SecondLevelModel(
    mask_img = group_glmmask).fit(
    cmap_list['p2p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}))
secondlvlglm_p2p1.compute_contrast('constant', output_type = 'z_score').to_filename(
    os.path.join(group_outdir, 'p2p1', 'group_p2p1_zmap.nii.gz'))
secondlvlglm_p2p1.compute_contrast('constant', output_type = 'effect_size').to_filename(
    os.path.join(group_outdir, 'p2p1', 'group_p2p1_beta.nii.gz'))

# non-parametric permutation on P2-P1
neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(
    cmap_list['p2p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}),
    second_level_contrast = "constant",
    n_perm=10000,
    two_sided_test = True,
    mask = group_glmmask,
    tfce = False,
    n_jobs = 6,
)
neg_log_pvals_permuted_ols_unmasked.to_filename(
    os.path.join(group_outdir, 'p2p1', 'group_p2p1_permute_neglogpval.nii.gz'))
math_img('img > -np.log10(0.05)', img = neg_log_pvals_permuted_ols_unmasked).to_filename(
    os.path.join(group_outdir, 'p2p1', 'group_p2p1_permute_neglogpval_05mask.nii.gz'))

# %%
# group statistic on P2-P1
if not os.path.exists(os.path.join(group_outdir, 'p3p1')):
    os.makedirs(os.path.join(group_outdir, 'p3p1'))
secondlvlglm_p3p1 = SecondLevelModel(
    mask_img = group_glmmask).fit(
    cmap_list['p3p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}))
secondlvlglm_p3p1.compute_contrast('constant', output_type = 'z_score').to_filename(
    os.path.join(group_outdir, 'p3p1', 'group_p3p1_zmap.nii.gz'))
secondlvlglm_p3p1.compute_contrast('constant', output_type = 'effect_size').to_filename(
    os.path.join(group_outdir, 'p3p1', 'group_p3p1_beta.nii.gz'))

# non-parametric permutation on P3-P1
neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(
    cmap_list['p3p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}),
    second_level_contrast = "constant",
    n_perm=10000,
    two_sided_test = True,
    mask = group_glmmask,
    tfce = False,
    n_jobs=6,
)
neg_log_pvals_permuted_ols_unmasked.to_filename(
    os.path.join(group_outdir, 'p3p1', 'group_p3p1_permute_neglogpval.nii.gz'))
math_img('img > -np.log10(0.05)', img = neg_log_pvals_permuted_ols_unmasked).to_filename(
    os.path.join(group_outdir, 'p3p1', 'group_p3p1_permute_neglogpval_05mask.nii.gz'))
# %%
# group statistic on P4-P1
if not os.path.exists(os.path.join(group_outdir, 'p4p1')):
    os.makedirs(os.path.join(group_outdir, 'p4p1'))
secondlvlglm_p4p1 = SecondLevelModel(
    mask_img = group_glmmask).fit(
    cmap_list['p4p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}))
secondlvlglm_p4p1.compute_contrast('constant', output_type = 'z_score').to_filename(
    os.path.join(group_outdir, 'p4p1', 'group_p4p1_zmap.nii.gz'))
secondlvlglm_p4p1.compute_contrast('constant', output_type = 'effect_size').to_filename(
    os.path.join(group_outdir, 'p4p1', 'group_p4p1_beta.nii.gz'))

# non-parametric permutation on P4-P1
neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(
    cmap_list['p4p1'],
    design_matrix = pd.DataFrame({'constant': np.ones(n_sub) * 1}),
    second_level_contrast = "constant",
    n_perm=10000,
    two_sided_test = True,
    mask = group_glmmask,
    tfce = False,
    n_jobs=6,
)
neg_log_pvals_permuted_ols_unmasked.to_filename(
    os.path.join(group_outdir, 'p4p1', 'group_p4p1_permute_neglogpval.nii.gz'))
math_img('img > -np.log10(0.05)', img = neg_log_pvals_permuted_ols_unmasked).to_filename(
    os.path.join(group_outdir, 'p4p1', 'group_p4p1_permute_neglogpval_05mask.nii.gz'))