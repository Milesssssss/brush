# Issue #7 synthetic depth comparison

Run from the repository root with Python 3, Rust 1.100.0 and a native GPU supported by Brush.

```sh
python3 crates/brush-bench-test/test_cases/issue7/generate.py target/issue7-depth
cargo +1.100.0 run --locked -p brush-process --no-default-features \
  --example issue7_depth_acceptance -- target/issue7-depth > target/issue7-depth/results.jsonl
```

The generator uses the Python standard library. Both COLMAP variants share the same 48 × 48 RGBA images, camera poses, masks and 300-point PLY. The runner uses the same six training views in a fixed rotation for 180 steps, with `view_00.png` and `view_04.png` held out.

| Case | Training supervision |
| --- | --- |
| `baseline` (twice) | RGB only. Training still reads the same TIFFs as in `metric`. |
| `metric` | Metric camera-Z TIFFs, depth loss weight 0.50. |
| `relative` | Relative inverse-depth TIFFs, depth loss weight 0.15. Each view uses a different inverse-depth offset. |

The runner writes one JSON row per case with held-out foreground PSNR, horizontal RGB gradient error, mean alpha outside the silhouette, coverage at alpha ≥ 0.25 and camera-Z MAE on covered pixels. The depth reference is the generator's metric TIFF for each held-out camera in every case. Compare depth MAE alongside coverage and use the two RGB runs to gauge variation.

To run a subset, pass case names after the fixture path, for example `target/issue7-depth baseline metric`. This test exercises sidecar loading and the training loss on known synthetic geometry. Gains from a learned depth prior on real captures, and the number of views needed for a target quality, require separate measurement.
