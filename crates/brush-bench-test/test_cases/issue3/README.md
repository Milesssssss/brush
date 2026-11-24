# Issue #3 synthetic acceptance

Run from the repository root with Python 3, Rust 1.100.0 and a native GPU supported by Brush.

```sh
python3 crates/brush-bench-test/test_cases/issue3/generate.py target/issue3-repro
cargo +1.100.0 run --locked -p brush-process --no-default-features \
  --example issue3_acceptance -- target/issue3-repro > target/issue3-repro/results.jsonl
```

The generator uses only the Python standard library. It writes four COLMAP variants with 48 × 48 RGBA images, TIFF sidecars, fixed camera poses, and the same 300-point initialization. Every case uses 180 steps with six training views in a fixed rotation and holds out `view_00.png` and `view_04.png`. For the exposure variants the held-out views retain the original colors.

| Cases | Comparison |
| --- | --- |
| `baseline` (twice), `metric`, `relative` | RGB alone, metric depth at weight 0.50, and relative inverse depth at weight 0.15. In the relative fixture, each TIFF uses a different offset of inverse depth. |
| `exposure_off`, `exposure_on` | Per-image affine compensation on paired data with biased training exposure. |
| `balanced_off`, `balanced_on` | The same comparison with approximately balanced exposure shifts. |
| `baseline`, `reset`, `reset_once` | Reset disabled, every 48 steps, or once at phase iteration 96. |
| `combined` | Biased exposure with affine compensation, metric depth and a 48-step reset interval. Saves and reloads a checkpoint after step 48. |

The runner writes one JSON row per case and checks that the views, visibility, split, reset and checkpoint paths are exercised. It reports held-out foreground PSNR, horizontal RGB gradient error, mean alpha outside the silhouette, foreground coverage at alpha ≥ 0.25, and camera-Z MAE for covered pixels. All depth errors use the same independent metric TIFFs. Compare depth MAE alongside coverage, and review the exposure and reset results alongside both PSNR and silhouette leakage. Use the duplicate RGB runs to gauge variation. Results can vary by GPU and backend.

Pass case names after the fixture path to run a subset, for example `target/issue3-repro baseline baseline metric relative`. See `--help` for the full list.
