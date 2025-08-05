#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Please input the subject number you would like to convert."
  exit 1
else
  for i in "$@"
  do
    docker run --user=$(id -u):$(id -g) -e "UID=$(id -u)" -e "GID=$(id -g)" --rm -t \
	    -v /work/O2Resting:/base \
	    nipy/heudiconv:latest \
	    -d /base/sourcedata/dicom/sub-{subject}/1/*/* \
	    -o /base/rawdata/ \
	    -f /base/code/step00_preparation/oxyfmri_heuristic.py \
	    -s $i \
	    -c dcm2niix \
	    -b --overwrite
    chmod 770 -R /work/O2Resting/rawdata/sub-$i
  done
fi
