from __future__ import annotations

from typing import Optional

from heudiconv.utils import SeqInfo


def create_key(
    template: Optional[str],
    outtype: tuple[str, ...] = ("nii.gz",),
    annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    t1w = create_key("sub-{subject}/anat/sub-{subject}_T1w")
    t2w = create_key("sub-{subject}/anat/sub-{subject}_T2w")
    boldmb = create_key("sub-{subject}/func/sub-{subject}_task-rest_run-{item:d}_bold")
    boldsb = create_key("sub-{subject}/func/sub-{subject}_task-rest_run-{item:d}_sbref") 
    fmapap = create_key("sub-{subject}/fmap/sub-{subject}_dir-AP_epi")
    fmappa = create_key("sub-{subject}/fmap/sub-{subject}_dir-PA_epi")
    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {
        t1w: [], t2w: [], boldmb: [], boldsb: [], fmapap: [], fmappa: []}

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """

        if (s.dim1 == 256) and (s.dim3 == 192) and ('NORM' in s.image_type) and ("T1w" in s.protocol_name):
            info[t1w].append(s.series_id)
        if (s.dim1 == 256) and (s.dim3 == 208) and ('NORM' in s.image_type) and ("T2w" in s.protocol_name):
            info[t2w].append(s.series_id)
        if (s.dim1 == 78) and (s.dim3 == 54) and (s.dim4 > 1) and ("ep2d_bold" in s.protocol_name):
            info[boldmb].append(s.series_id)
        if (s.dim1 == 78) and (s.dim3 == 54) and (s.dim4 == 1) and ("ep2d_bold" in s.protocol_name):
            info[boldsb].append(s.series_id)
        if (s.dim1 == 78) and (s.dim3 == 54) and ("FieldMap_PA" in s.protocol_name):
            info[fmappa].append(s.series_id)
        if (s.dim1 == 78) and (s.dim3 == 54) and ("FieldMap_AP" in s.protocol_name):
            info[fmapap].append(s.series_id)
    return info
