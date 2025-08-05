#!/bin/bash
docker run --user=$(id -u):$(id -g) -e "UID=$(id -u)" -e "GID=$(id -g)" --rm -t \
    -v /work/O2Resting/rawdata:/data:ro \
    -v /work/O2Resting/derivatives:/out \
    -v /work/O2Resting/temporary:/work \
    nipreps/mriqc:latest \
    -w /work \
    --no-sub \
    --fd_thres 0.5 \
    /data /out/mriqc \
    group
chmod 770 -R /work/O2Resting/derivatives/mriqc/
