# %%
import os
import pandas as pd
import json
import re
import readCMRRPhysio
import numpy as np
import json
import matplotlib.pyplot as plt
# %%
rootpath = "/work/O2Resting/"
sourcepath = os.path.join(rootpath, "sourcedata/dicom")
rawdatapath = os.path.join(rootpath, "rawdata")
physiooutpath = os.path.join(rootpath, "sourcedata/physio")
# %%
subjinfo = pd.read_csv(os.path.join(rawdatapath, "participants.tsv"), sep = "\t")
subjlist = [_ for _ in subjinfo["participant_id"] if re.match("sub[-]\d\d\d", _)]
# %%
for i_sub in subjlist:
    sub_id = i_sub.split('-')[1]#"001"
    subjoutdir = os.path.join(rawdatapath, f"sub-{sub_id}")
    for i_run in [1, 2]:
        func_json_path = os.path.join(subjoutdir, "func",
                                        f"sub-{sub_id}_task-rest_run-{i_run}_bold.json")
        with open(func_json_path) as json_txt:
            func_json = json.load(json_txt)
            
        func_physio_num = func_json["SeriesNumber"] + 1
        func_nscans = len(func_json['time']['samples']['AcquisitionNumber'])
        func_tr = func_json['RepetitionTime']
        # %%
        physio_dir = os.path.join(sourcepath, f"sub-{sub_id}/1/{func_physio_num}")
        physio_filename = os.listdir(physio_dir)[0]
        physio_path = os.path.join(sourcepath, f"sub-{sub_id}/1/{func_physio_num}",
                                    physio_filename)

        # %% 
        # output the log file
        physio_tempdir = os.path.join(f"./temp/sub-{sub_id}_run-{i_run}")
        if not os.path.exists(physio_tempdir):
            os.makedirs(physio_tempdir)
        physio_outdir = os.path.join(physiooutpath, f"sub-{sub_id}_run-{i_run}")
        if not os.path.exists(physio_outdir):
            os.makedirs(physio_outdir)
        physio_dat = readCMRRPhysio.readCMRRPhysio(physio_path, 0, physio_tempdir)
        
        physio_tempfiles = os.listdir(physio_tempdir)
        FIND_INFO_LOG = False
        FIND_PULS_LOG = False
        FIND_RESP_LOG = False
        for tempfile in physio_tempfiles:
            # reformate and output
            with open(os.path.join(physio_tempdir, tempfile), encoding = "ascii", mode = "r") as f:
                    logtxt = f.readline().split("\\n")
                
            with open(
                os.path.join(physio_outdir, tempfile),
                "w") as outf:
                for line in logtxt:
                    if re.match("^b'(.*)", line):
                        outtxt = re.match("^b'(.*)", line).group(1)
                        outf.write(f"{outtxt}\n")
                    else:
                        outf.write(f"{line}\n")
        
            # read info log
            if re.match("^.*_Info.log", tempfile):
                FIND_INFO_LOG = True
                with open(os.path.join(physio_tempdir, tempfile), encoding = "ascii", mode = "r") as f:
                    logtxt = f.readline().split("\\n")
                
                logtable = list()
                for line in logtxt:
                    reoutput = re.search(
                        "^ +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) *$", line)
                    if reoutput:
                        logtable.append(
                            [reoutput.group(1), reoutput.group(2), 
                            reoutput.group(3), reoutput.group(4), 
                            reoutput.group(5)])
                log_info_table = pd.DataFrame(logtable, 
                    columns = ["VOLUMN", "SLICE", "ACQ_START_TICS", "ACQ_END_TICS", "ECHO"])
                
            # read puls log
            if re.match("^.*_PULS.log", tempfile):
                FIND_PULS_LOG = True
                with open(os.path.join(physio_tempdir, tempfile), encoding = "ascii", mode = "r") as f:
                    logtxt = f.readline().split("\\n")
                
                logtable = list()
                for line in logtxt:
                    if re.search("^SampleTime += +(\d+) *$", line):
                        log_puls_sampletime = re.search("^SampleTime += +(\d+) *$", line).group(1)
                    reoutput = re.search(
                        "^ +(\d+) +([A-Z]+) +(\d+) +([A-Z_]*) *$", line)
                    if reoutput:
                        logtable.append(
                            [reoutput.group(1), reoutput.group(2), 
                            reoutput.group(3), reoutput.group(4)])
                log_puls_table = pd.DataFrame(logtable, 
                    columns = ["ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"])
            
            if re.match("^.*_RESP.log", tempfile):
                FIND_RESP_LOG = True
                with open(os.path.join(physio_tempdir, tempfile), encoding = "ascii", mode = "r") as f:
                    logtxt = f.readline().split("\\n")
                
                logtable = list()
                for line in logtxt:
                    if re.search("^SampleTime += +(\d+) *$", line):
                        log_resp_sampletime = re.search("^SampleTime += +(\d+) *$", line).group(1)
                    reoutput = re.search(
                        "^ +(\d+) +([A-Z]+) +(\d+) +([A-Z_]*) *$", line)
                    if reoutput:
                        logtable.append(
                            [reoutput.group(1), reoutput.group(2), 
                            reoutput.group(3), reoutput.group(4)])
                log_resp_table = pd.DataFrame(logtable, 
                    columns = ["ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"])
                
        # %%
        # read the frequence, i.e. the "ticks", (should be 400 Hz or 1 tick per 2.5ms)
        physio_freq = physio_dat['Freq']
        ticktime = 1/physio_dat["Freq"]
        # get timestamp from info, mark the timestamp that start acqusition
        starttick = log_info_table[
            (log_info_table["VOLUMN"] == "0") & (log_info_table["SLICE"] == "0")
            ]["ACQ_START_TICS"].astype(int)[0]
        # get timestamp from puls, change the timestamp to onset
        if FIND_PULS_LOG:
            log_puls_table["Onset"] = np.round((
                log_puls_table["ACQ_TIME_TICS"].astype(int) - starttick) * ticktime,
                                               decimals = 4)
            log_puls_table = log_puls_table[["Onset", "ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"]]
            log_puls_table.to_csv(
                os.path.join(
                    subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_recording-pulse_physio.tsv.gz'),
                sep = "\t", index = False, compression = 'gzip')
            
            log_puls_outjson = dict({
                "SamplingFrequency": physio_freq/int(log_puls_sampletime),
                "StartTime": log_puls_table["Onset"][0],
                "Columns": ["Onset", "ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"],
            })
            with open(os.path.join(
                subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_recording-pulse_physio.json'), "w") as outfile:
                json.dump(log_puls_outjson, outfile)
            
        if FIND_RESP_LOG:
            log_resp_table["Onset"] = np.round((
                log_resp_table["ACQ_TIME_TICS"].astype(int) - starttick) * ticktime,
                                               decimals = 4)
            log_resp_table = log_resp_table[["Onset", "ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"]]
            log_resp_table.to_csv(
                os.path.join(
                    subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_recording-respiration_physio.tsv.gz'),
                sep = "\t", index = False, compression = 'gzip')

            log_resp_outjson = dict({
                "SamplingFrequency": physio_freq/int(log_resp_sampletime),
                "StartTime": log_resp_table["Onset"][0],
                "Columns": ["Onset", "ACQ_TIME_TICS", "CHANNEL", "VALUE", "SIGNAL"],
            })
            with open(os.path.join(
                subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_recording-respiration_physio.json'), "w") as outfile:
                json.dump(log_resp_outjson, outfile)
        
       
        # the index of when scanner start and end to record the scans
        start_idx = np.min(np.where(physio_dat['ACQ'])[0])
        end_idx = np.max(np.where(physio_dat['ACQ'])[0])

        # the onset in seconds
        onset = np.round((
            np.arange(len(physio_dat['ACQ']))  - start_idx)/physio_dat['Freq'], 
                         decimals = 4)

        # prepare data frame
        physio_df = pd.DataFrame({
            'Onset': onset,
        })
        if 'PULS' in physio_dat.keys():
            physio_df['Pulse'] = physio_dat["PULS"]
        if 'RESP' in physio_dat.keys():
            physio_df['Respiration'] = physio_dat["RESP"]

        # %%
        physio_df.to_csv(
            os.path.join(
                subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_physio.tsv.gz'),
            sep = "\t", index = False, compression = 'gzip')
        physio_outjson = dict({
            "SamplingFrequency": physio_freq,
            "StartTime": onset[0],
            "Columns": ["Onset"],
        })
        if 'PULS' in physio_dat.keys():
            physio_outjson['Columns'].append('Pulse')
        if 'RESP' in physio_dat.keys():
            physio_outjson['Columns'].append('Respiration')
        with open(os.path.join(
                subjoutdir, 'func', f'sub-{sub_id}_task-rest_run-{i_run}_physio.json'), "w") as outfile:
            json.dump(physio_outjson, outfile)

