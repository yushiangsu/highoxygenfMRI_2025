# %%
import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# %% customized ppg preprocessing
# add artifacts correction in ppg_peaks
def ppg_preproc_pipe(ppg_raw, sampling_rate):
    # clean signal
    ppg_cleaned  = nk.ppg_clean(
         ppg_raw, sampling_rate = sampling_rate, method = 'nabian2018')
    
    # find peaks
    ppg_peaksig, ppg_info = nk.ppg_peaks(
        ppg_cleaned, sampling_rate = sampling_rate, method = 'elgendi',
        correct_artifacts = True, show = False)

    ppg_info['sampling_rate'] = sampling_rate

    # rate computation
    ppg_rate = nk.signal_rate(
        ppg_info['PPG_Peaks'], sampling_rate = sampling_rate, 
        desired_length = len(ppg_cleaned))

    # wrapped up
    ppg_sig = pd.DataFrame(
        {
            'PPG_Raw': ppg_raw,
            'PPG_Clean': ppg_cleaned,
            'PPG_Rate': ppg_rate,
            'PPG_Peaks': ppg_peaksig['PPG_Peaks'].values,
    })
    
    return ppg_sig, ppg_info

# 
def rsp_preproc_pipe(rsp_raw, sampling_rate):
    # clean signal
    rsp_cleaned = nk.rsp_clean(
        rsp_raw, sampling_rate = sampling_rate,
        method = "hampel", threshold = 3, window_length = 2) #0.25*sampling_rate)
    rsp_cleaned = nk.signal.signal_filter(
        rsp_cleaned, sampling_rate = sampling_rate, method = 'savgol', 
        window_size = 1 * sampling_rate)

    # find peaks
    rsp_peaksig, rsp_info = nk.rsp_peaks(
        rsp_cleaned, sampling_rate = sampling_rate,
        method = "scipy", peak_distance = 2, peak_prominence = 0.5)
    rsp_info["sampling_rate"] = sampling_rate
    
    # Get additional parameters
    rsp_phase = nk.rsp_phase(rsp_peaksig, desired_length=len(rsp_cleaned))
    rsp_amplitude = nk.rsp_amplitude(rsp_cleaned, rsp_peaksig)
    rsp_rate = nk.signal_rate(
        rsp_info["RSP_Troughs"], sampling_rate = sampling_rate,
        desired_length = len(rsp_cleaned))
    rsp_symmetry = nk.rsp_symmetry(rsp_cleaned, rsp_peaksig)
    rsp_rvt = nk.rsp_rvt(
        rsp_cleaned, method = "harrison2021",
        sampling_rate = sampling_rate, silent=True)
    
    # wrapped up
    rsp_sig = pd.DataFrame(
        {
            'RSP_Raw': rsp_raw,
            'RSP_Clean': rsp_cleaned,
            'RSP_Rate': rsp_rate,
            'RSP_Amplitude': rsp_amplitude,
            'RSP_Rate': rsp_rate,
            'RSP_RVT': rsp_rvt,
    })
    rsp_sig = pd.concat([rsp_sig, rsp_phase, rsp_symmetry, rsp_peaksig], axis=1)
    
    return rsp_sig, rsp_info
    
# %% prepare data path
prjdir = "/work/O2Resting/"
datdir = os.path.join(prjdir, "rawdata")
subinfo = pd.read_csv(
    os.path.join(prjdir, "rawdata/participants.tsv"), sep = "\t")
n_sub = subinfo.shape[0]

outdir = os.path.join(prjdir, "derivatives/neurokit_physio/")
if not os.path.exists(outdir):
    os.makedirs(outdir)
# %%
for i_sub in subinfo["participant_id"]:
    print(f"=========working on subject: {i_sub}===========")
    subdir = os.path.join(datdir, i_sub, 'func')
    suboutdir = os.path.join(outdir, i_sub)
    if not os.path.exists(suboutdir):
        os.makedirs(suboutdir)
    for i_run in [1, 2]:
        ##
        ## PPG
        ##
        ppg_file = os.path.join(
            subdir, f"{i_sub}_task-rest_run-{i_run}_recording-pulse_physio.tsv.gz")
        if os.path.exists(ppg_file):
            ppg_dt = pd.read_csv(ppg_file, sep = "\t")
            ppg_t = np.round(ppg_dt['Onset'].values, decimals = 4)
            ppg_sr = 1/np.median(np.round(np.diff(ppg_t), decimals = 4))
            ppg_ts = ppg_dt['VALUE'].values
            if ppg_sr != 200.0:
                print(f"{i_sub} ppg sampling rate is not 200.")
            
            # resample data, somtimes it missed a tic
            ppg_nt  = np.round(np.arange(ppg_t[0], ppg_t[-1], 1/ppg_sr), decimals=4)
            ppg_nts = np.interp(ppg_nt, ppg_t, ppg_ts)
            
            # get hr
            ppg_sig, ppg_info = ppg_preproc_pipe(
                ppg_nts, sampling_rate = ppg_sr)
            ppg_sig = pd.concat(
                [pd.DataFrame(ppg_nt, columns = ["Onset"]), ppg_sig], 
                axis = 1)
            
            # calculate moving variability
            win_len = 6.0 # 6 seconds
            ppg_hrv = list()
            for i_t in ppg_sig['Onset'].values:
                ppg_hrv.append(np.var(ppg_sig['PPG_Rate'][
                    (ppg_sig['Onset'] > i_t - win_len/2) & 
                    (ppg_sig['Onset'] <= i_t + win_len/2)]))
            ppg_sig['PPG_HRV'] = ppg_hrv
            
            # plot and save
            nk.ppg_plot(ppg_sig, ppg_info)
            fig = plt.gcf()
            fig.set_dpi(300)
            fig.set_size_inches(16, 10)
            fig.savefig(os.path.join(suboutdir, f"{i_sub}_run-{i_run}_recording-pulse_physio.png"),
                        dpi = 300)
            
            ## resample
            ppg_newt = np.arange(-8, ppg_nt[-1]+0.5, 0.5)
            ppg_dspd = list()
            for i_col  in ppg_sig.columns:
                ppg_dspd.append(np.interp(ppg_newt,  ppg_nt, ppg_sig[i_col]))
            ppg_dspd = pd.DataFrame(np.array(ppg_dspd).T, 
                                    columns = ppg_sig.columns)
            

            # save data
            ppg_dspd.round(decimals = 4).to_csv(
                os.path.join(suboutdir, f"{i_sub}_run-{i_run}_recording-pulse_physio.tsv"),
                sep = "\t", index = False)
        
        #
        # RSP
        #
        rsp_file = os.path.join(
            subdir, f"{i_sub}_task-rest_run-{i_run}_recording-respiration_physio.tsv.gz")
        if os.path.exists(rsp_file):
            rsp_dt = pd.read_csv(rsp_file, sep = "\t")
            rsp_t = np.round(rsp_dt['Onset'].values, decimals = 4)
            rsp_sr = 1/np.median(np.round(np.diff(rsp_t), decimals = 4))
            rsp_ts = rsp_dt['VALUE'].values
            if rsp_sr != 50.0:
                print(f"{i_sub} rsp sampling rate is not 50.")
            
            # resample data, somtimes it missed a tic
            rsp_nt  = np.round(np.arange(rsp_t[0], rsp_t[-1], 1/rsp_sr), decimals=4)
            rsp_nts = np.interp(rsp_nt, rsp_t, rsp_ts)
            
            # ignore if there is no signal changes
            if np.sum(np.diff(rsp_ts) == 0)/len(rsp_ts) < 0.95:
                rsp_nts = nk.standardize(rsp_nts)
                rsp_sig, rsp_info = rsp_preproc_pipe(
                    rsp_nts, sampling_rate = rsp_sr)
                rsp_sig = pd.concat(
                    [pd.DataFrame(rsp_nt, columns = ["Onset"]), 
                    rsp_sig], 
                    axis = 1)
                
                # RV and RRV
                win_len = 6.0 # 6 seconds
                rsp_rv = list()
                rsp_rrv = list()
                for i_t in rsp_sig['Onset'].values:
                    rsp_rv.append(np.var(rsp_sig['RSP_Clean'][
                        (rsp_sig['Onset'] > i_t - win_len/2) & 
                        (rsp_sig['Onset'] <= i_t + win_len/2)]))
                    rsp_rrv.append(np.var(rsp_sig['RSP_Rate'][
                        (rsp_sig['Onset'] > i_t - win_len/2) & 
                        (rsp_sig['Onset'] <= i_t + win_len/2)]))
                rsp_sig['RSP_RV'] = rsp_rv
                rsp_sig['RSP_RRV'] = rsp_rrv
                
                # Plotting
                nk.rsp_plot(rsp_sig, rsp_info)
                fig = plt.gcf()
                fig.set_dpi(300)
                fig.set_size_inches(16, 24)
                fig.savefig(os.path.join(suboutdir, f"{i_sub}_run-{i_run}_recording-respiration_physio.png"),
                            dpi = 300)
                
                ## resample
                rsp_newt = np.arange(-8, rsp_nt[-1]+0.5, 0.5)
                rsp_dspd = list()
                for i_col  in rsp_sig.columns:
                    rsp_dspd.append(np.interp(rsp_newt, rsp_nt, rsp_sig[i_col]))
                rsp_dspd = pd.DataFrame(np.array(rsp_dspd).T, 
                                        columns = rsp_sig.columns)
            
                # save data
                rsp_dspd.round(decimals = 4).to_csv(
                    os.path.join(suboutdir, f"{i_sub}_run-{i_run}_recording-respiration_physio.tsv"),
                    sep = "\t", index = False)
        