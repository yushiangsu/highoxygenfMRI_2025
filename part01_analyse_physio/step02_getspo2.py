## 
## During the final batch of data collection, we were finally able 
## to record the SpO2 signal. At that point, SpO2 data was recorded 
## from only seven participants, and there were still some technical 
## issues with the incomplete recordings. As a result, the SpO2 data  
## is incomplete and was used only for validation purposes; it is not 
## presented in the manuscript.
##
# %%
import os
import re
import numpy as np
import pandas as pd
import neurokit2 as nk

# %% prepare data path
prjdir = "/work/O2Resting/"
spo2_dir = os.path.join(prjdir, "sourcedata", "spo2")
datdir = os.path.join(prjdir, "rawdata")
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
n_sub = subinfo.shape[0]
outdir = "/work/O2Resting/derivatives/spo2/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
# %%
for i_sub in subinfo["participant_id"]:
    subid = i_sub.split('-')[1]
    if os.path.exists(os.path.join(spo2_dir, subid)):
        sub_outdir = os.path.join(outdir, i_sub)
        for i_run in [1, 2]:
            spo2_file = os.path.join(
                spo2_dir, subid, f'sub-{subid}_run-{i_run}_spo2.txt')
            if os.path.exists(spo2_file):
                with open(spo2_file, 'r') as f:
                    spo2_text = f.read()
            if isinstance(spo2_text, str):
                spo2_text = spo2_text.split('\n')
            spo2_dat = list()
            for i_row in spo2_text:
                rematch = re.match(r'(\d+\.\d\d\d\d),(.+)', i_row)
                if rematch:
                    spo2_dat.append([rematch.group(1), rematch.group(2)])
            
            # some text is missing, trying to fix it...
            spo2_fixdat = list()
            for i_dat in spo2_dat:
                refindval = re.search(r'(\d+)\sHR',i_dat[1])
                if refindval:
                    if len(refindval.group(1)) > 1:
                        spo2_fixdat.append([i_dat[0], refindval.group(1)])
                    else:
                        spo2_fixdat.append([i_dat[0], '9'+refindval.group(1)])
                else:
                    spo2_fixdat.append([i_dat[0], None])
                    
            spo2_pd = pd.DataFrame(np.array(spo2_fixdat), columns = ["Onset", "SpO2"])
            if sum(spo2_pd['SpO2'].isna()) < 0.25*spo2_pd.shape[0]:
                spo2_pd['SpO2_fillmissing'] = nk.signal.signal_fillmissing(spo2_pd['SpO2'])
                if not os.path.exists(sub_outdir):
                    os.makedirs(sub_outdir)
                spo2_pd.to_csv(
                    os.path.join(sub_outdir, f'{i_sub}_run-{i_run}_recording-spo2_physio.tsv'),
                    sep = "\t", index = False)
        
# %%
