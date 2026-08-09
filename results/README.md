# Result indexes

This directory tracks compact `*.summary.json` indexes only. They make each
reported result's scope, frozen configuration, acceptance rule and evidence
location machine-readable without redistributing restricted data, weights,
TIFFs or per-file metrics.

An index must state whether its result is independently recomputable from the
Git tree. Do not put raw experiment outputs here; those remain in the ignored
`experiments/` or `outputs/` directories.
