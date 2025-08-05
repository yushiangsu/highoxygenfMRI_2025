#!/bin/bash
datpath=/work/O2Resting/derivatives/fmriprep
icapath=/work/O2Resting/derivatives/melodic_ica
spname=T1w
if [ $# -eq 0 ]; then
  echo "Please input the subject number you would like to do preprocessing."
  exit 1
else
  for i in "$@"
  do
    echo "Start to run Melodic ICA on subject: ${i}"
    # BET T1w images
    echo "===> get BET T1w image"
    bet ${datpath}/sub-${i}/anat/sub-${i}_desc-preproc_T1w.nii.gz \
        ${icapath}/sub-${i}/sub-${i}_desc-bet_T1w.nii.gz \
        -R -f 0.5 -g 0

    for irun in 1 2
    do
      echo "===> Working on run: ${irun}"
      # maks out bold and boldref
      echo "=======> get BET BOLD images"
      fslmaths ${datpath}/sub-${i}/func/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-preproc_bold.nii.gz \
          -mas ${datpath}/sub-${i}/func/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-brain_mask.nii.gz \
          ${icapath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-bet_bold.nii.gz
      fslmaths ${datpath}/sub-${i}/func/sub-${i}_task-rest_run-${irun}_space-${spname}_boldref.nii.gz \
          -mas ${datpath}/sub-${i}/func/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-brain_mask.nii.gz \
          ${icapath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-bet_boldref.nii.gz

      # run MELODIC ICA
      echo "=======> run MELODIC ICA"
      feat  ${icapath}/sub-${i}/sub-${i}_run-${irun}_space-${spname}_melodicica_design.fsf

      # copy motion parameter
      mkdir -p ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/mc/
      cp ${icapath}/sub-${i}/run-${irun}_motion.par \
          ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/mc/prefiltered_func_data_mcf.par

      # run fix
      echo "=======> run ICA FIX"
      fix -f ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica
      fix -c ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica Standard 20
    done
  done
fi
