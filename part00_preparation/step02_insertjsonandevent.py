import sys
import os
import json
import pandas as pd

data_dir = '/work/O2Resting/'
subj_list = sys.argv[1:]

for i_subj in subj_list:
    rawdatadir = os.path.join(data_dir, 'rawdata', f'sub-{i_subj}')

    #### write two echo time information to fieldmap images

    if os.path.exists(os.path.join(rawdatadir, 'fmap', f'sub-{i_subj}_dir-AP_epi.json')):
        with open(os.path.join(rawdatadir, 'fmap', f'sub-{i_subj}_dir-AP_epi.json')) as f:
            ap_json = json.load(f)
        with open(os.path.join(rawdatadir, 'fmap', f'sub-{i_subj}_dir-PA_epi.json')) as f:
            pa_json = json.load(f)
    
        # write "IntendedFor" information to json
        # The fieldmaps were intended to correct task EPI
        task_filename_list = list()
        for i_run in range(2):
            if os.path.exists(os.path.join(rawdatadir, 'func', f"sub-{i_subj}_task-rest_run-{i_run+1}_sbref.nii.gz")):
                task_filename_list.append(f"func/sub-{i_subj}_task-rest_run-{i_run+1}_sbref.nii.gz")
            elif os.path.exists(os.path.join(rawdatadir, 'func', f"sub-{i_subj}_task-rest_run-{i_run+1}_sbref.nii")):
                task_filename_list.append(f"func/sub-{i_subj}_task-rest_run-{i_run+1}_sbref.nii")
            if os.path.exists(os.path.join(rawdatadir, 'func', f"sub-{i_subj}_task-rest_run-{i_run+1}_bold.nii.gz")):
                task_filename_list.append(f"func/sub-{i_subj}_task-rest_run-{i_run+1}_bold.nii.gz")
            elif os.path.exists(os.path.join(rawdatadir, 'func', f"sub-{i_subj}_task-rest_run-{i_run+1}_bold.nii")):
                task_filename_list.append(f"func/sub-{i_subj}_task-rest_run-{i_run+1}_bold.nii")
            
        ap_json['IntendedFor'] = task_filename_list
        pa_json['IntendedFor'] = task_filename_list

        with open(os.path.join(rawdatadir, 'fmap', f'sub-{i_subj}_dir-AP_epi.json'), 'w') as f:
            json.dump(ap_json, f, indent=4, sort_keys=True)
        with open(os.path.join(rawdatadir, 'fmap', f'sub-{i_subj}_dir-PA_epi.json'), 'w') as f:
            json.dump(pa_json, f, indent=4, sort_keys=True)
    else:
        print(f'{i_subj} did not have fmap images')

    #### write event information
    event_pd = pd.DataFrame({
        'onset': [0.0, 300.0, 600.0],
        'duration': [300.0, 300.0, 600.0],
        'trial_type': ['inhale_air', 'inhale_oxygen', 'inhale_air_again'],
    })
    for i_run in range(2):
        if os.path.exists(os.path.join(rawdatadir, 'func', f'sub-{i_subj}_task-rest_run-{i_run+1}_bold.json')):
            event_pd.to_csv(
                os.path.join(rawdatadir, 'func', f'sub-{i_subj}_task-rest_run-{i_run+1}_events.tsv'),
                sep = "\t", index = False
                )
        else:
            print(f'{i_subj} did not have func-run{i_run+1} images')