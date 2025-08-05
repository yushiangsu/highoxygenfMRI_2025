#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Please input the subject number you would like to do preprocessing."
  exit 1
else
  cp $FREESURFER_HOME/license.txt /work/O2Resting/temporary/
  for i in "$@"
  do
    docker run --user=$(id -u):$(id -g) -e "UID=$(id -u)" -e "GID=$(id -g)" --rm -t \
        -v /work/O2Resting/rawdata:/data:ro \
        -v /work/O2Resting/derivatives:/out \
        -v /work/O2Resting/temporary:/work \
        nipreps/fmriprep:24.0.1 \
        -w /work \
        --participant_label $i \
        --fs-license-file /work/license.txt \
        --output-spaces MNI152NLin2009cAsym anat fsaverage \
        --slice-time-ref 0 \
        /data /out/fmriprep \
        participant
    chmod 770 -R /work/O2Resting/rawdata/sub-$i
  done
fi

