# Training parameters: how to tune them

CLI flags answer *what* a setting does; this guide covers *how to adjust them* for training length, final splat count, and VRAM. Run `brush --help` for defaults and the full flag list.

## Goals and the main knobs

| Goal | Start here |
| --- | --- |
| Shorter / longer runs | `--total-train-iters`, `--growth-stop-iter` |
| Fewer / more splats | `--max-splats`, `--growth-grad-threshold`, `--growth-select-fraction`, `--refine-every` |
| Lower VRAM | `--max-resolution`, `--max-splats`, `--sh-degree`, `--max-scene-batch-cache-size` |
| Faster iteration on a big dataset | `--subsample-frames`, `--subsample-points`, `--max-frames` |

Defaults target a full-quality run (~30k steps, generous splat budget). For a quick preview, lower resolution and iterations first; only then tighten growth.

## Training length

### `--total-train-iters` (default `30000`)

Total optimization steps. Quality usually keeps improving for a while after growth stops, so cutting this short mainly saves wall time.

- **Quick look / debug:** `2000`–`5000`
- **Solid result:** `15000`–`30000`
- **Long refine after growth:** keep the default (or higher) once growth has stopped

Also lower `--growth-stop-iter` when you shorten the run, or growth may still be active near the end.

### `--growth-stop-iter` (default `15000`)

After this step, densification stops (clamped to `total_train_iters`). Remaining steps refine existing splats.

- Short runs: set near `total_train_iters * 0.5`
- If the cloud is still under-covering the scene at the stop point, raise this (and usually `total_train_iters`)
- If you already have enough splats and only want polishing, lower it

### `--refine-every` (default `200`)

How often densify/prune runs. Rough rule: about the number of images needed to cover the scene once.

- Fewer images → try a lower value (more frequent refine)
- Huge datasets → a slightly higher value reduces refine overhead
- Very low values grow the cloud faster (and use VRAM sooner)

## Splat count

`--max-splats` is a **ceiling**, not a target. Actual count comes from growth + pruning while under that limit.

### `--max-splats` (default `10000000`)

Hard upper bound. Also used to subsample oversized SfM init clouds at load time.

- **Low VRAM / mobile:** `500000`–`2000000`
- **Typical desktop:** `2000000`–`5000000`
- Only raise toward the default if you still see holes after growth stops and you have headroom

If training hits the cap early, quality plateaus: raise `--max-splats` or slow growth (higher `--growth-grad-threshold`, lower `--growth-select-fraction`).

### `--growth-grad-threshold` (default `0.0025`)

Lower → more splats marked to grow. Raise to grow slower / stay smaller.

### `--growth-select-fraction` (default `0.25`)

Fraction of “needs growth” candidates that actually split/clone. Raise for more aggressive growth; lower to stay compact.

### `--opac-decay` (default `0.004`)

Gently pushes opacity down so weak splats can be pruned. Slightly higher decay can keep counts down; too high and you lose coverage.

Skipped for the whole run when `--opacity-reset-every` is enabled. The decay is a slow stand-in for a reset; applying both would stack two downward pressures. Leave the reset at `0` to keep this decay.

### `--opacity-reset-every` (default `0`)

Opt-in periodic opacity reset. `0` leaves training unchanged, including `--opac-decay`.

When set to `N > 0`, every `N` phase iterations (not at step 0, and not after 95% of the phase — the same cutoff as refine) each splat's opacity is clamped to `min(a, 0.01)` and the Adam moments for opacity are zeroed. The cap is the opacity after the 3D-filter compensation when a scale floor is attached, and the raw parameter is solved so that compensated value hits the cap. Splats already at or below `0.01` are left unchanged, so a faint splat is not brightened, and the reset does not bake scales. The Adam step counter is kept, so the first updates after a reset stay small instead of immediately walking opacity back up.

`N` does not have to be a multiple of `--refine-every`. On a step that also refines, the reset runs after prune and split. While the reset is on, `--opac-decay` is not applied.

## VRAM and memory

VRAM scales mainly with **resolution × splat count × SH degree**, plus the dataset image cache.

### `--max-resolution` (default `1920`)

Long-edge cap for training views. The highest-impact VRAM lever after splat count.

- **Tight VRAM:** `800`–`1280`
- **Balanced:** `1280`–`1600`
- **Full quality:** `1920` (or higher if images are larger and you have room)

Start low to find good growth settings, then raise resolution for a final run.

### `--sh-degree` (default `3`)

Spherical-harmonics degree (0–4). Higher → richer view-dependent color and more memory/compute per splat.

- OOM or slow: try `2` or `1`
- Mostly diffuse scenes: `1`–`2` is often enough

### `--max-scene-batch-cache-size` (default ~`6GiB` native / `2GiB` wasm)

CPU-side cache for decoded training frames. Lower if the machine is RAM-limited (`2GiB`, `1GiB`); raise on large datasets if I/O is the bottleneck. This is system RAM, not GPU VRAM, but contention can still cause pressure.

### `--subsample-frames` / `--max-frames` / `--subsample-points`

Fewer views and fewer SfM points mean less work per step and a smaller starting cloud.

- Use `--subsample-frames 2` or `4` to scout settings
- `--subsample-points` helps when the init PLY is huge (also see `--max-splats` subsampling at load)

## Depth supervision

Depth loss is disabled by default. Supply float32 TIFF sidecars from an external source, then set a positive depth loss weight.

| Flag | Default | What it does |
| --- | --- | --- |
| `--depth-loss-weight` | `0` | Weight of depth supervision against COLMAP `depth/*.tiff`. `0` keeps the four-channel color renderer. |
| `--depth-mode` | `metric` | `metric` compares inverse camera Z. `relative-inverse` uses masked patch and image-scale normalization on unitless, near-high inverse depth. |
| `--estimate-metric-scale` | off | One global scale from those depth maps and projected landmarks. Skipped when no depth maps are present. `--units-per-meter` still applies afterwards. |

Match each single-channel float32 TIFF to its RGB stem (`images/frame.png` → `depth/frame.tiff`). Metric samples must be finite and positive. `--estimate-metric-scale` uses the metric map's own resolution and scales whichever initializer is used, including an overriding PLY. Values and camera positions are divided by `--units-per-meter` before the metric loss.

Relative inverse depth accepts finite signed values and uses non-finite values to mark invalid pixels. Leave the TIFF values in their original unitless scale. The loss uses 11×11 patches, a normalized residual margin of 0.2, and local and image-scale terms weighted 0.1 and 1.0 before applying `--depth-loss-weight`. The source alpha mask weights the statistics and the residuals. Relative-inverse mode excludes pixels with no rendered depth and requires metric scale estimation to be off.

The [synthetic depth comparison](../crates/brush-bench-test/test_cases/issue7/README.md) uses the same initialization and view split for RGB, metric depth, and relative depth. To measure the benefit of an offline predictor on captures, keep the same SfM initialization and real held-out images across the three runs and evaluate each training-view count separately.

## Learning rates and loss (usually leave alone)

`--lr-mean`, `--lr-mean-end`, `--lr-coeffs-dc`, `--lr-opac`, `--lr-scale`, `--lr-rotation`, `--ssim-weight`, and related flags match common Gaussian-splatting defaults. `--depth-loss-weight` stays `0` unless the scene has depth maps you want to supervise with.

Touch these only when diagnosing divergence or known scene quirks. Prefer resolution, growth, and iteration budget for normal tuning.

## Process / export knobs (time and disk, not model size)

| Flag | Practical note |
| --- | --- |
| `--eval-every` | Lower for more frequent metrics (slower). Raise for long overnight runs. |
| `--eval-split-every` | Hold out every Nth view for eval. Needed for meaningful PSNR/SSIM. |
| `--eval-save-to-disk` | Writes eval renders under the export path; useful for visual QA, costs disk I/O. |
| `--export-every` / `--export-path` | Checkpoint cadence vs disk use. |

## Suggested recipes

**Fast preview (low VRAM)**

```text
brush <dataset> --total-train-iters 4000 --growth-stop-iter 2500 \
  --max-resolution 800 --max-splats 1500000 --sh-degree 2
```

**Balanced desktop run**

```text
brush <dataset> --total-train-iters 20000 --growth-stop-iter 12000 \
  --max-resolution 1280 --max-splats 3000000
```

**Quality run (enough VRAM)**

```text
brush <dataset> --total-train-iters 30000 --growth-stop-iter 15000 \
  --max-resolution 1920 --max-splats 5000000 --sh-degree 3
```

If you OOM: lower `--max-resolution`, then `--max-splats`, then `--sh-degree`. If the cloud looks sparse after growth stops: lower `--growth-grad-threshold` or raise `--growth-select-fraction` / `--growth-stop-iter` before maxing out splat budget.
