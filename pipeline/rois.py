"""vertex-level predictions -> ROI timeseries.

STUB. The current ROI reduction lives inline in
``extract.preds_to_roi_timeseries``. If ROI handling grows (extra regions,
per-hemisphere splits, smoothing), lift it here and keep it leaning on the
TRIBE repo's ``utils_fmri`` — do NOT hand-roll vertex->region mapping.

Cortical ROIs only (fsaverage5): vmPFC, anterior insula. Nucleus accumbens is
subcortical and absent from this output.
"""
