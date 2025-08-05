#!/bin/bash
icapath=/work/O2Resting/derivatives/melodic_ica
fixoutpath=/work/O2Resting/derivatives/fix_agg_blankcomps
spname=T1w
if [ $# -eq 0 ]; then
  echo "Please input the subject number you would like to do preprocessing."
  exit 1
else
  for i in "$@"
  do
    echo "Start to apply on subject: ${i}"
    
    for irun in 1 2
    do
      echo "===> Working on run: ${irun}"
      # apply fix
      fix -a ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/blankfix.txt -A

      # move file to another path
      mkdir -p ${fixoutpath}/sub-${i}/
      mv ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/filtered_func_data_clean.nii.gz \
        ${fixoutpath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-denoise_bold.nii.gz
      #mv ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/filtered_func_data_clean_vn.nii.gz \
      #  ${fixoutpath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-denoise_variancenormalization.nii.gz
      cp ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/mask.nii.gz \
        ${fixoutpath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-denoise_mask.nii.gz
      cp ${icapath}/sub-${i}/space-${spname}_run-${irun}.ica/example_func.nii.gz \
        ${fixoutpath}/sub-${i}/sub-${i}_task-rest_run-${irun}_space-${spname}_desc-denoise_boldref.nii.gz
    done
  done
fi
