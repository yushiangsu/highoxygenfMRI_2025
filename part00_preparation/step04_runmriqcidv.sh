#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Please input the subject number you would like to do preprocessing."
  exit 1
else
  for i in "$@"
  do
    docker run --user=$(id -u):$(id -g) -e "UID=$(id -u)" -e "GID=$(id -g)" --rm -t \
        -v /work/O2Resting/rawdata:/data:ro \
        -v /work/O2Resting/derivatives:/out \
        -v /work/O2Resting/temporary:/work \
        nipreps/mriqc:latest \
        -w /work \
        --participant_label $i \
        --no-sub \
        --fd_thres 0.5 \
        /data /out/mriqc \
        participant
    chmod 770 -R /work/O2Resting/derivatives/mriqc/sub-$i
  done
fi

