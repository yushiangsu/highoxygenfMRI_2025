library(tidyverse)
library(emmeans)
library(effectsize)
library(lme4)
library(lmerTest)

#### loading information ####
dat_path = "../../../derivatives/neurokit_physio/"
spo2_dat_path = "../../../derivatives/spo2/"
subjinfo = read.csv("../../../rawdata/participants.tsv", sep = "\t")
onset_list = c(0.0, 300.0, 600.0, 900.0)

#### get data for all physio metrics ####
physio_all = data.frame()
for (i_sub in subjinfo$participant_id) {
  for (i_run in c(1,2)) {
    # get heart rate and 6s moving average HRV from timeseries data
    pulsts_tsv_path = sprintf(
      "%s%s/%s_run-%d_recording-pulse_physio.tsv", dat_path, i_sub, i_sub, i_run)
    
    puls_HR = rep(NA, length(onset_list))
    puls_6smvHRV = rep(NA, length(onset_list))
    if (file.exists(pulsts_tsv_path)) {
      puls_dat = read.csv(pulsts_tsv_path, sep = "\t")
      for (i_onset in 1:length(onset_list)) {
        # get index
        if (i_onset != length(onset_list)) {
          pick_idx = (puls_dat['Onset'] >= onset_list[i_onset]) & (puls_dat['Onset'] < onset_list[i_onset + 1])
        } else {
          pick_idx = (puls_dat['Onset'] >= onset_list[i_onset])
        }
        
        # if have valid data
        if (any(pick_idx)) {
          puls_HR[i_onset] = mean(puls_dat['PPG_Rate'][pick_idx])
          puls_6smvHRV[i_onset] = mean(puls_dat['PPG_HRV'][pick_idx])
        } else {
          print(sprintf('%s, run-%d has no valid data for onset %.1f', i_sub, i_run, i_onset))
        }
       
      }
    } else {
      cat(sprintf("%s, run-%s did not find the pulse file\n", i_sub, i_run))
    }
    
    # get breath rate and 6s moving average RV and RRV from timeseries data
    respts_tsv_path = sprintf(
      "%s%s/%s_run-%d_recording-respiration_physio.tsv", dat_path, i_sub, i_sub, i_run)
    
    resp_BR = rep(NA, length(onset_list))
    resp_RVT = rep(NA, length(onset_list))
    resp_6smvRRV = rep(NA, length(onset_list))
    if (file.exists(respts_tsv_path)) {
      resp_dat = read.csv(respts_tsv_path, sep = "\t")
      
      for (i_onset in 1:length(onset_list)) {
        # get index
        if (i_onset != length(onset_list)) {
          pick_idx = (resp_dat['Onset'] >= onset_list[i_onset]) & (resp_dat['Onset'] < onset_list[i_onset + 1])
        } else {
          pick_idx = (resp_dat['Onset'] >= onset_list[i_onset])
        }
        
        # if have valid data
        if (any(pick_idx)) {
          resp_BR[i_onset] = mean(resp_dat['RSP_Rate'][pick_idx])
          resp_RVT[i_onset] = mean(resp_dat['RSP_RVT'][pick_idx])
          resp_6smvRRV[i_onset] = mean(resp_dat['RSP_RRV'][pick_idx])
        } else {
          print(sprintf('%s, run-%d has no valid data for onset %.1f', i_sub, i_run, i_onset))
        }
        
      }
    } else {
      cat(sprintf("%s, run-%s did not find the resp file\n", i_sub, i_run))
    }
    
    # get SpO2 timeseries data
    spo2ts_tsv_path = sprintf(
      "%s%s/%s_run-%d_recording-spo2_physio.tsv", spo2_dat_path, i_sub, i_sub, i_run)
    
    spo2_spo2 = rep(NA, length(onset_list))
    if (file.exists(spo2ts_tsv_path)) {
      spo2_dat = read.csv(spo2ts_tsv_path, sep = "\t")
      for (i_onset in 1:length(onset_list)) {
        # get index
        if (i_onset != length(onset_list)) {
          pick_idx = (spo2_dat['Onset'] >= onset_list[i_onset]) & (spo2_dat['Onset'] < onset_list[i_onset + 1])
        } else {
          pick_idx = (spo2_dat['Onset'] >= onset_list[i_onset])
        }
        
        # if have valid data
        if (any(pick_idx)) {
          spo2_spo2[i_onset] = mean(spo2_dat['SpO2_fillmissing'][pick_idx])
        } else {
          print(sprintf('%s, run-%d has no valid data for onset %.1f', i_sub, i_run, i_onset))
        }
      }
    } else {
      cat(sprintf("%s, run-%s did not find the SpO2 file\n", i_sub, i_run))
    }
    
    # organize all metrics here
    physio_sub = data.frame(
      subid = rep(i_sub, length(onset_list)),
      run = rep(i_run, length(onset_list)),
      onset = onset_list,
      
      puls_HR = puls_HR,
      puls_6smvHRV = puls_6smvHRV,
      resp_BR = resp_BR,
      resp_RVT = resp_RVT,
      resp_6smvRRV = resp_6smvRRV,
      spo2 = spo2_spo2)
    
    # combined with big data
    physio_all = rbind(physio_all, physio_sub)
  } # end i_run
} # end i_sub

#### output data ####
write.csv(physio_all, "./physio_all.csv", row.names = F)

#### analysis ####
physio_all = physio_all %>%
  mutate(subid = factor(subid), run = factor(run), onset = factor(onset)) %>%
  mutate(num_run = case_when(run == 1 ~  0.5, run == 2 ~ -0.5))

## HR
aovmd00 = aov(puls_HR ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd00)
eta_squared(aovmd00)
# post hoc
aovmd00_emm = emmeans(aovmd00, "onset", data = physio_all)
pairs(aovmd00_emm)
t_to_eta2(t = c(5.551, 3.639, 3.384, -1.913, -2.168, -0.255), df_error = 57)
# plot 5x12 in
lmmd00 = lmer(puls_HR ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd00, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit

ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(puls_HR)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  scale_y_continuous(limits = c(48, 100), breaks = c(50, 60, 70, 80, 90)) +
  xlab("Duration (minutes)") +
  ylab("Heartrate") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))

## HRV
aovmd01 = aov(puls_6smvHRV ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd01)
eta_squared(aovmd01)
# plot 5x12 in
lmmd01 = lmer(puls_6smvHRV ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd01, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit

ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(puls_6smvHRV)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  scale_y_continuous(limits = c(-0.5, 40), breaks = c(0, 10, 20, 30)) +
  xlab("Duration (minutes)") +
  ylab("HRV") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))

## RR
aovmd02 = aov(resp_BR ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd02)
eta_squared(aovmd02)
# plot 5x12 in
lmmd02 = lmer(resp_BR ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd02, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit
ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(resp_BR)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  scale_y_continuous(limits = c(7, 23), breaks = c(10, 15, 20)) +
  xlab("Duration (minutes)") +
  ylab("Respiratory Rate") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))

## RVT
aovmd03 = aov(resp_RVT ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd03)
eta_squared(aovmd03)
# plot 5x12 in
lmmd03 = lmer(resp_RVT ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd03, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit

ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(resp_RVT)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  scale_y_continuous(limits = c(0.2, 1.35), breaks = c(0.2, 0.4, 0.6, 0.8, 1.0)) +
  xlab("Duration (minutes)") +
  ylab("Respiratory volume over time (RVT)") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))

## RRV
aovmd04 = aov(resp_6smvRRV ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd04)
eta_squared(aovmd04)
# plot 5x12 in
lmmd04 = lmer(resp_6smvRRV ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd04, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit
ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(resp_6smvRRV)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  scale_y_continuous(limits = c(0, 8), breaks = c(0, 2, 4, 6)) +
  xlab("Duration (minutes)") +
  ylab("RRV") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))

## SpO2
aovmd05 = aov(spo2 ~ onset * run + Error(subid/(onset*run)), 
              data = physio_all)
summary(aovmd05)
eta_squared(aovmd05)
# plot 5x12 in
lmmd05 = lmer(spo2 ~ onset * num_run + (1 | subid), data = physio_all)
newdat = expand.grid(onset = unique(physio_all$onset),
                     num_run = 0)
newdat_pred = predict(lmmd05, newdata = newdat, re.form = NA, se.fit = T)
newdat$Y = newdat_pred$fit
newdat$Ymin = newdat_pred$fit - newdat_pred$se.fit
newdat$Ymax = newdat_pred$fit + newdat_pred$se.fit

ggplot(data = physio_all %>% group_by(subid, onset) %>% summarise(meanY = mean(spo2)),
       aes(x = onset, y = meanY)) +
  geom_violin(color = 'gray80', fill = 'gray80') +
  geom_line(aes(group = subid), color = 'gray60') +
  geom_pointrange(data = newdat, aes(y = Y, ymin = Ymin, ymax = Ymax), 
                  color = 'gray20', size = 2, linewidth = 3) +
  xlab("Duration (minutes)") +
  ylab("SpO2") +
  theme(axis.title = element_text(size = 24), axis.text = element_text(size = 18),
        plot.title = element_text(size = 30))