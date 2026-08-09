# Local asset manifest

Raw data, checkpoints and TIFF results are deliberately not versioned. Before
unpacking a download, copy `manifest.local.example.json` to
`manifest.local.json`, enter the archive filename and SHA-256 published by the
source record, then run:

```bash
python scripts/doctor.py --assets
```

The production check expects these paths after extraction:

- `example/checkpoints/fluoresfm/epoch_0_iter_700000.pt`
- `example/checkpoints/biomedclip/open_clip_config.json`
- `example/checkpoints/biomedclip/open_clip_pytorch_model.bin`
- `example/data/BioSR_MT/test/channel_0/`

`manifest.local.json` is ignored because it may contain private download notes;
the example template is tracked so its schema is reviewable.
