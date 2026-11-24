use std::f32::consts::FRAC_1_SQRT_2;
use std::path::{Path, PathBuf};

use crate::{
    adam_scaled::{AdamScaled, AdamState},
    appearance::{Appearance, IDENTITY_WEIGHT, correct_image},
    config::TrainConfig,
    msg::{RefineStats, TrainStepStats},
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    splat_init::bounds_from_pos,
    split::{MIN_OPACITY, MIN_SPLIT_K, SPLIT_OPACITY_EPS},
    stats::RefineRecord,
};
use brush_dataset::{config::DepthMode, scene::SceneBatch};
use brush_loss::{
    ImageLossConfig, expected_depth, image_loss, inverse_depth_l1, relative_inverse_depth_loss,
};
use brush_render::bwd::render_splats_with_mode;
use brush_render::gaussian_splats::{RasterPass, RasterizationMode, Splats};
use brush_render::{AlphaMode, bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
use burn::{
    module::Param,
    tensor::{
        Bool, Device, Distribution, Gradients, IndexingUpdateOp::Assign, Int, Tensor, TensorData,
        activation::sigmoid, s,
    },
};

use hashbrown::HashSet;
use rand::SeedableRng;
use tracing::{Instrument, trace_span};

pub const BOUND_PERCENTILE: f32 = 0.8;

/// Fraction of a training phase after which refine and the opacity reset both
/// stop, so the tail of the phase can settle.
pub const REFINE_STOP_PROGRESS: f32 = 0.95;

/// Opacity cap used by [`SplatTrainer::reset_opacity`].
/// Applied to the opacity the renderer uses: `sigmoid(raw)` times the
/// 3D-filter energy compensation when a scale floor is attached.
pub const OPACITY_RESET_MAX: f32 = 0.01;

/// Fraction of training after which the Mip-Splatting 3D-filter floor stops
/// being recomputed and is held frozen (still applied), so splats settle
/// against a fixed target instead of chasing a moving floor.
const MIN_SCALE_FREEZE_FRAC: f32 = 0.9;

/// The three per-parameter Adam states of a [`Splats`] module, owned directly
/// so the trainer can update LR scaling every step and surgically edit the
/// momentum tensors during refine — all GPU-side, no record round-trips.
pub(crate) struct SplatOptim {
    pub(crate) adam: AdamScaled,
    pub(crate) transforms: AdamState<2>,
    pub(crate) sh_coeffs: AdamState<3>,
    pub(crate) opacities: AdamState<1>,
}

impl SplatOptim {
    /// Fresh optimizer state for a splat module of the given SH degree.
    /// Momentum starts empty; the transforms LR scaling is attached by the
    /// trainer each step.
    pub(crate) fn new(config: &TrainConfig, sh_degree: u32, opt_device: &Device) -> Self {
        let num_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

        // DC (band 0) uses full LR; bands 1+ are scaled down.
        let mut scales = vec![1.0f32; num_coeffs];
        let rest_scale = 1.0 / config.lr_coeffs_sh_scale;
        for s in &mut scales[1..] {
            *s = rest_scale;
        }
        let sh_lr_scales = Tensor::<1>::from_floats(scales.as_slice(), opt_device).reshape([
            1,
            num_coeffs as i32,
            1,
        ]);

        Self {
            adam: AdamScaled::new(1e-15),
            transforms: AdamState::new(None, false),
            sh_coeffs: AdamState::new(Some(sh_lr_scales), true),
            opacities: AdamState::new(None, false),
        }
    }
}

/// Step one parameter: pull its gradient, run Adam on the inner
/// (autodiff-free) tensor, and re-wrap tracking. Parameters without a
/// gradient this step are left untouched.
fn step_param<const D: usize>(
    adam: &AdamScaled,
    lr: f64,
    param: Param<Tensor<D>>,
    state: &mut AdamState<D>,
    grads: &mut Gradients,
    grad_sq_mean: Option<Tensor<D>>,
) -> Param<Tensor<D>> {
    param.map(|t| {
        let Some(grad) = t.grad_remove(grads) else {
            return t;
        };
        let stepped = adam.step(lr, t.inner(), &grad, grad_sq_mean, state);
        Tensor::from_inner(stepped).require_grad()
    })
}

pub struct SplatTrainer {
    config: TrainConfig,
    /// Per-step multiplier of the exponential mean-LR schedule:
    /// `lr(n) = lr_mean * decay^(n-1)`.
    lr_mean_decay: f64,
    /// Per-column LR scales for `transforms` (`means(3) + rotations(4) +
    /// log_scales(3)`) with the mean columns zeroed, and a mask of those
    /// columns: the scheduled mean LR is mixed in on the device each step so
    /// the optimizer never waits on a host upload.
    lr_scaling_fixed: Tensor<2>,
    lr_mean_columns: Tensor<2>,
    refine_record: Option<RefineRecord>,
    pub(crate) optim: Option<SplatOptim>,
    pub(crate) appearance: Option<Appearance>,
    ssim_enabled: bool,
    pub(crate) bounds: BoundingBox,
    pub(crate) step_count: u32,
    pub(crate) max_sh_degree: u32,
    rng: rand::rngs::StdRng,
    /// Per-train-view (world center, focal in px at native res) for the
    /// Mip-Splatting 3D filter. Empty disables it. The floor itself lives on
    /// the splats (recomputed at each refine), not here.
    view_cams: Vec<(glam::Vec3, f32)>,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel>,
}

fn inv_sigmoid(x: Tensor<1>) -> Tensor<1> {
    (x.clone() / (1.0f32 - x)).log()
}

/// Host-side `inv_sigmoid(OPACITY_RESET_MAX)`. Every clamped splat is written
/// to this exact value, so the cap does not depend on a per-element GPU log.
fn opacity_reset_raw_value() -> f32 {
    (OPACITY_RESET_MAX / (1.0 - OPACITY_RESET_MAX)).ln()
}

/// `min(sigmoid(raw), OPACITY_RESET_MAX)`, written back through `inv_sigmoid`.
/// Rows already at or below the cap keep their raw bits (no sigmoid round-trip).
fn clamp_raw_opacity(raw: Tensor<1>) -> Tensor<1> {
    let above = sigmoid(raw.clone()).greater_elem(OPACITY_RESET_MAX);
    let reset_raw = Tensor::zeros(raw.dims(), &raw.device()).add_scalar(opacity_reset_raw_value());
    raw.mask_where(above, reset_raw)
}

/// Energy compensation `Π s / sqrt(s² + f²)`, the same product `fold_min_scale`
/// folds into opacity. `transforms` is `[N, 10]`, `min_scale` is `[N]`.
fn opacity_floor_coef(transforms: &Tensor<2>, min_scale: &Tensor<1>) -> Tensor<1> {
    let n = transforms.dims()[0] as i32;
    let min_scale = min_scale.clone().to_device(&transforms.device());
    let log_scales = transforms.clone().slice(s![.., 7..10]);
    let s2 = log_scales.clone().mul_scalar(2.0).exp();
    let f2 = min_scale.clone().mul(min_scale).reshape([n, 1]);
    let new_log = s2.add(f2).log().mul_scalar(0.5);
    log_scales.sub(new_log).sum_dim(1).exp().reshape([n])
}

/// Cap the filter-compensated opacity at [`OPACITY_RESET_MAX`] and solve for
/// the raw parameter: `sigmoid(raw') * coef = min(sigmoid(raw) * coef, cap)`.
/// Rows whose compensated opacity is already at or below the cap keep their
/// raw bits, so a faint splat is not brightened. Scales and the floor stay put.
fn clamp_raw_opacity_filtered(
    raw: Tensor<1>,
    transforms: &Tensor<2>,
    min_scale: &Tensor<1>,
) -> Tensor<1> {
    let coef = opacity_floor_coef(transforms, min_scale);
    let effective = sigmoid(raw.clone()).mul(coef.clone());
    let above = effective.greater_elem(OPACITY_RESET_MAX);
    // `coef > cap` whenever `effective > cap` (effective ≤ coef), so the
    // target sigmoid `cap / coef` is in (0, 1). The clamp only guards rows
    // the mask does not write.
    let target = coef.clamp_min(1e-12).recip().mul_scalar(OPACITY_RESET_MAX);
    let target_raw = inv_sigmoid(target.clamp(1e-6, 1.0 - 1e-6));
    raw.mask_where(above, target_raw)
}

/// Same bookends as the refine gate in the training loop: not at iteration 0,
/// only on multiples of `every`, and not in the last 5% of the phase.
/// `every == 0` is off.
fn opacity_reset_due(phase_iter: u32, phase_progress: f32, every: u32) -> bool {
    every > 0
        && phase_iter > 0
        && phase_iter.is_multiple_of(every)
        && phase_progress <= REFINE_STOP_PROGRESS
}

/// Per-splat world-space scale floor for the Mip-Splatting 3D filter:
/// `f_i = sqrt(factor) · min_v(||mean_i - cam_v|| / focal_px_v)`. `means` and
/// the result are on the inner (non-autodiff) backend; `f` is a frozen
/// constant. Returns `None` if disabled or there are no cameras.
fn compute_min_scale(
    means: &Tensor<2>,
    view_cams: &[(glam::Vec3, f32)],
    factor: f32,
) -> Option<Tensor<1>> {
    if factor <= 0.0 || view_cams.is_empty() {
        return None;
    }
    let device = means.device();
    let n = means.dims()[0] as i32;

    let mut min_ratio: Option<Tensor<1>> = None;
    for (center, focal) in view_cams {
        let c = Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let diff = means.clone() - c;
        let dist = diff.clone().mul(diff).sum_dim(1).sqrt().reshape([n]);
        let ratio = dist.div_scalar(focal.max(1e-6));
        min_ratio = Some(match min_ratio {
            Some(m) => m.min_pair(ratio),
            None => ratio,
        });
    }
    min_ratio.map(|r| r.mul_scalar(factor.sqrt()))
}

pub async fn get_splat_bounds(splats: Splats, percentile: f32) -> BoundingBox {
    let means: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .try_to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    pub fn new(config: &TrainConfig, device: &Device, bounds: BoundingBox) -> Self {
        Self::new_seeded(config, device, bounds, 42)
    }

    pub fn new_seeded(
        config: &TrainConfig,
        device: &Device,
        bounds: BoundingBox,
        seed: u64,
    ) -> Self {
        // The per-step decay reaching lr_mean_end at the last iteration. With
        // one iteration or fewer there is nothing to decay over (and the
        // exponent 1/iters would be undefined), so hold the LR.
        let decay = if config.total_train_iters > 1 {
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64)
        } else {
            1.0
        };

        let ssim_enabled = config.ssim_weight > 0.0;

        // Optimizer state lives on the inner device.
        let opt_device = device.clone().inner();
        let (rot, scale) = (config.lr_rotation as f32, config.lr_scale as f32);
        let lr_scaling_fixed = Tensor::<1>::from_floats(
            [0.0, 0.0, 0.0, rot, rot, rot, rot, scale, scale, scale],
            &opt_device,
        )
        .reshape([1, 10]);
        let lr_mean_columns = Tensor::<1>::from_floats(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &opt_device,
        )
        .reshape([1, 10]);

        // Growth is gated on the global iter. LOD phases run past
        // total_train_iters but their refines should never grow — clamp
        // here so growth_stop is never effectively past end-of-training,
        // and growth_start never past growth_stop.
        let mut config = config.clone();
        config.growth_stop_iter = config.growth_stop_iter.min(config.total_train_iters);
        config.growth_start_iter = config.growth_start_iter.min(config.growth_stop_iter);

        #[cfg(not(target_family = "wasm"))]
        let lpips = (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device));

        Self {
            config,
            lr_mean_decay: decay,
            lr_scaling_fixed,
            lr_mean_columns,
            optim: None,
            appearance: None,
            refine_record: None,
            ssim_enabled,
            bounds,
            step_count: 0,
            max_sh_degree: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            view_cams: Vec::new(),
            #[cfg(not(target_family = "wasm"))]
            lpips,
        }
    }

    /// Percentile bounding box of the splats, refreshed on each refine.
    pub fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    /// Whether this phase iteration should clamp opacity and zero its Adam moments.
    ///
    /// Follows the refine schedule: skipped at phase iteration 0, only when
    /// `phase_iter` is a multiple of [`TrainConfig::opacity_reset_every`], and
    /// not once the phase passes [`REFINE_STOP_PROGRESS`]. `N` does not have
    /// to be a multiple of `refine_every`; the training loop applies the reset
    /// after prune/split on steps where both are due.
    pub fn opacity_reset_due(&self, phase_iter: u32, phase_progress: f32) -> bool {
        opacity_reset_due(phase_iter, phase_progress, self.config.opacity_reset_every)
    }

    /// Clamp opacity to at most [`OPACITY_RESET_MAX`] and replace the opacity
    /// Adam moments with zeros.
    ///
    /// The cap is the opacity after the 3D-filter energy compensation when a
    /// scale floor is attached (`sigmoid(raw) * coef`). The raw parameter is
    /// solved so the compensated value lands on the cap; scales and the floor
    /// are not baked. Splats already at or below the cap are left unchanged.
    /// Without a floor this is `min(sigmoid(raw), cap)`.
    ///
    /// The Adam step counter is left unchanged. That matches replacing the
    /// parameter tensor in the optimizer: `exp_avg` / `exp_avg_sq` are cleared
    /// and the bias-correction step is kept, so the next updates do not
    /// immediately undo the clamp. Moments for means, scales, rotation, and
    /// SH are not touched. A missing optimizer (no step yet) still clamps the
    /// parameters; the next step then starts from fresh moments.
    pub fn reset_opacity(&mut self, mut splats: Splats) -> Splats {
        splats.raw_opacities = if let Some(floor) = splats.min_scale.clone() {
            let transforms = splats.transforms.val();
            splats
                .raw_opacities
                .map(|raw| clamp_raw_opacity_filtered(raw, &transforms, &floor))
        } else {
            splats.raw_opacities.map(clamp_raw_opacity)
        };
        if let Some(optim) = self.optim.as_mut() {
            optim
                .opacities
                .map_momentum(|moment| Tensor::zeros(moment.dims(), &moment.device()));
        }
        splats
    }

    /// Supply per-train-view (world center, focal-px at native res) to enable
    /// the Mip-Splatting 3D filter (gated on `config.min_scale_factor > 0`).
    pub fn set_view_cams(&mut self, view_cams: Vec<(glam::Vec3, f32)>) {
        self.view_cams = view_cams;
    }

    /// Bind appearance rows to this dataset's training view order.
    pub fn set_appearance_views(
        &mut self,
        paths: impl IntoIterator<Item = PathBuf>,
        device: &Device,
    ) {
        if self.config.appearance_compensation {
            let paths = paths.into_iter().collect();
            if let Some(appearance) = &mut self.appearance {
                appearance.set_views(paths);
            } else {
                self.appearance = Some(Appearance::new(paths, &device.clone().inner()));
            }
        }
    }

    /// Keep view corrections and their optimizer when restarting the splat optimizer for a LOD.
    pub fn carry_appearance_from(&mut self, previous: &mut Self) {
        self.appearance = previous.appearance.take();
    }

    /// The learned affine residual for the dataset-relative image path.
    pub fn appearance_for_view(&self, path: &Path) -> Option<Tensor<2>> {
        self.appearance.as_ref()?.for_view(path)
    }

    /// Install refine statistics without a training step.
    ///
    /// Tests use this to drive split/prune from known screen sizes. Production
    /// fills the same record from `step`.
    #[cfg(test)]
    pub(crate) fn testing_set_refine_record(&mut self, record: crate::stats::RefineRecord) {
        self.refine_record = Some(record);
    }

    /// Drop the GPU-side optimizer, refine and appearance state ahead of a
    /// memory compaction. Save a checkpoint first if the state must survive.
    pub fn release_gpu_state(&mut self) {
        self.optim = None;
        self.refine_record = None;
        self.appearance = None;
    }

    pub async fn step(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        let mut splats = splats;

        // Track max SH degree from the first splats we see.
        if self.step_count == 0 {
            self.max_sh_degree = splats.sh_degree();
        }
        self.step_count += 1;

        let [img_h, img_w] = batch.img_size();
        let camera = batch.camera;

        let device = splats.device();
        let has_alpha = batch.has_alpha;
        // GT lives on the GPU as packed `[H, W]` u32 (RGBA u8). All mixing
        // (bg compositing, alpha matching, mask) is folded into the loss
        // kernels; no f32 GT image is ever materialised here.
        // GT is pure data — never differentiated. Build it on the inner
        // backend so it doesn't inherit the autodiff device's residual
        // checkpointing flag (the LPIPS `unpack_gt_rgb` path, via
        // `unwrap_wgpu_int`, expects a clean Wgpu tensor).
        let gt_packed: Tensor<2, Int> =
            Tensor::from_data(batch.img_packed, &device.clone().inner());
        let img_size = glam::uvec2(img_w as u32, img_h as u32);
        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(
            base_bg,
            self.config.background_noise_strength,
            &mut self.rng,
        );

        let median_scale = self.bounds.median_size();

        let lr_mean = self.config.lr_mean
            * self.lr_mean_decay.powi(self.step_count as i32 - 1)
            * median_scale as f64;
        let masked_alpha = batch.alpha_mode == AlphaMode::Masked;
        let do_alpha_match = has_alpha && !masked_alpha && self.config.match_alpha_weight > 0.0;

        // The optimizer runs with base LR 1 and these per-column scales.
        let lr_scaling =
            self.lr_scaling_fixed.clone() + self.lr_mean_columns.clone() * lr_mean as f32;
        let appearance_delta = self
            .appearance
            .as_ref()
            .map(|appearance| appearance.for_training(batch.view_index));

        let (
            mut grads,
            visible,
            opacities,
            num_visible,
            loss_inner,
            refine_weight,
            max_radius,
            coeffs_grad_sq,
        ) = {
            // The splats already carry their 3D-filter floor (set at refine);
            // the render path folds it in. Optimizer/refine work on raw params.
            let render_input = splats.clone();
            // Depth stays off unless this view has a map and the weight is
            // positive, so the colour render and its gradients are unchanged.
            let use_depth = batch.depth.is_some() && self.config.depth_loss_weight > 0.0;
            let raster_mode = if use_depth {
                RasterizationMode::RgbaAndDepth
            } else {
                RasterizationMode::Rgba
            };
            let diff_out = render_splats_with_mode(
                render_input,
                &camera,
                img_size,
                background,
                RasterPass::Backward,
                raster_mode,
            )
            .instrument(trace_span!("Forward"))
            .await;

            let pred_image = diff_out.img;
            let refine_weight_holder = diff_out.refine_weight_holder;
            let visible = diff_out.visible;
            let max_radius = diff_out.max_radius;
            let opacities = diff_out.opacities;

            // RGB loss is `(1 - w) * L1 + (-w) * SSIM` per pixel. Bg
            // compositing always runs in the kernel; for synthesised opaque
            // alpha or zero bg it's a no-op. Mask multiplies the loss-map
            // by `gt.a`; for synthesised opaque alpha that's a no-op too.
            // Alpha matching needs a real alpha source (synthesised
            // a = 1 would pull predicted alpha to fully opaque); we feed
            // `pred` with 4 channels and the kernel's `c == 3` workgroup
            // emits `|pred.a - gt.a|` into the alpha channel.
            let (l1_w, ssim_w) = if self.ssim_enabled {
                (1.0 - self.config.ssim_weight, -self.config.ssim_weight)
            } else {
                (1.0, 0.0)
            };
            // Only composite when there's a real alpha channel and a non-zero
            // bg to mix in; the kernel skips the per-pixel `(1-a)*bg` math
            // entirely when this is None.
            let composite_bg = (has_alpha && background != glam::Vec3::ZERO).then_some(background);
            let cfg = ImageLossConfig {
                l1_weight: l1_w,
                ssim_weight: ssim_w,
                composite_bg,
                mask: masked_alpha,
                alpha_weight: if do_alpha_match {
                    self.config.match_alpha_weight
                } else {
                    0.0
                },
            };
            // Depth renders append a 5th channel. The image loss still sees
            // only RGBA, and appearance affects RGB without touching depth.
            let pred_for_loss = if use_depth {
                pred_image.clone().slice(s![.., .., 0..4])
            } else {
                pred_image.clone()
            };
            let pred_for_loss = match &appearance_delta {
                Some(delta) => correct_image(pred_for_loss, delta.clone(), background),
                None => pred_for_loss,
            };
            let mut loss = image_loss(pred_for_loss.clone(), gt_packed.clone(), cfg);
            if let Some(delta) = &appearance_delta {
                loss = loss + delta.clone().powi_scalar(2).mean() * IDENTITY_WEIGHT;
            }

            // LPIPS still needs an f32 RGB tensor for VGG. Materialising it
            // here costs ~99 MB at 4K, only when LPIPS is enabled.
            #[cfg(not(target_family = "wasm"))]
            if let Some(lpips) = &self.lpips {
                let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
                let gt_rgb_diff: Tensor<3> = Tensor::from_inner(gt_rgb);
                loss = loss
                    + lpips.lpips(
                        pred_for_loss.slice(s![.., .., 0..3]).unsqueeze_dim(0),
                        gt_rgb_diff.unsqueeze_dim(0),
                    ) * self.config.lpips_loss_weight;
            }

            if use_depth && let Some(depth_data) = batch.depth {
                let gt_depth = Tensor::<2>::from_data(depth_data.data, &device);
                let accumulated = pred_image.clone().slice(s![.., .., 4..5]);
                // Detach alpha in the quotient. The rasterizer also leaves the
                // depth channel out of the alpha dot-accumulator, so depth
                // error cannot be reduced by changing opacity.
                let alpha = pred_image.clone().slice(s![.., .., 3..4]);
                let rendered = expected_depth(accumulated, alpha).reshape([img_h, img_w]);
                // Masked views already ignore RGB outside the silhouette. Depth
                // uses the same alpha byte as a per-pixel weight.
                let depth_mask = masked_alpha.then(|| {
                    let alpha_byte = gt_packed
                        .clone()
                        .bitwise_right_shift_scalar(24)
                        .bitwise_and_scalar(0xFF)
                        .float()
                        .div_scalar(255.0);
                    Tensor::from_inner(alpha_byte)
                });
                let depth_loss = match depth_data.mode {
                    DepthMode::Metric => inverse_depth_l1(rendered, gt_depth, depth_mask),
                    DepthMode::RelativeInverse => {
                        relative_inverse_depth_loss(rendered, gt_depth, depth_mask)
                    }
                };
                loss = loss + depth_loss * self.config.depth_loss_weight;
            }

            // Strip the autodiff graph off the loss so consumers can read the
            // scalar later without keeping the backward pass alive.
            let loss_inner = loss.clone().inner();
            let mut grads = splats.bwd_validate(loss).await;

            let refine_weight = refine_weight_holder
                .grad_remove(&mut grads)
                .expect("XY gradients need to be calculated.")
                .without_autodiff();
            // Reduced in the backward off the compact rows, so the dense SH
            // gradient never gets squared just to be summed away.
            let coeffs_grad_sq = diff_out
                .coeffs_grad_sq_holder
                .grad_remove(&mut grads)
                .map(Tensor::without_autodiff);

            (
                grads,
                visible,
                opacities,
                diff_out.num_visible,
                loss_inner,
                refine_weight,
                max_radius,
                coeffs_grad_sq,
            )
        };

        // The optimizer strips autodiff before stepping, so optimizer state
        // (scaling, momentum) lives on the inner device.
        let opt_device = device.clone().inner();
        let optimizer = self
            .optim
            .get_or_insert_with(|| SplatOptim::new(&self.config, splats.sh_degree(), &opt_device));

        optimizer.transforms.scaling = Some(lr_scaling);

        splats = trace_span!("Optimizer step").in_scope(|| {
            splats.transforms = trace_span!("Transforms step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    1.0,
                    splats.transforms,
                    &mut optimizer.transforms,
                    &mut grads,
                    None,
                )
            });
            splats.sh_coeffs = trace_span!("SH Coeffs step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    self.config.lr_coeffs_dc,
                    splats.sh_coeffs,
                    &mut optimizer.sh_coeffs,
                    &mut grads,
                    coeffs_grad_sq,
                )
            });
            splats.raw_opacities = trace_span!("Opacity step").in_scope(|| {
                step_param(
                    &optimizer.adam,
                    self.config.lr_opac,
                    splats.raw_opacities,
                    &mut optimizer.opacities,
                    &mut grads,
                    None,
                )
            });
            splats
        });

        if let (Some(appearance), Some(delta)) = (&mut self.appearance, appearance_delta) {
            appearance.update(batch.view_index, delta, &mut grads);
        }

        trace_span!("Housekeeping").in_scope(|| {
            // Refine state accumulates on the inner (non-autodiff) device.
            // Kept after the optimizer so it doesn't sit between the
            // backward's gradient gathers and the step that consumes them.
            let device = splats.device().inner();
            let record = self
                .refine_record
                .get_or_insert_with(|| RefineRecord::new(splats.num_splats(), &device));
            record.gather_stats(refine_weight, visible.clone(), max_radius);
        });

        // Noise uses the forward's floored opacities, without building an autodiff graph.
        let samples = Tensor::random(
            [splats.num_splats() as usize, 3],
            Distribution::Normal(0.0, 1.0),
            &splats.device().inner(),
        );

        splats.transforms = splats.transforms.map(|t| {
            let inner = t.inner();
            // Resolve random generation and views before the arithmetic so the
            // gate through means addition can fuse as one rank-2 expression.
            let means = inner.clone().slice(s![.., 0..3]);
            let opacities = opacities.unsqueeze_dim::<2>(1);
            let visible = visible.unsqueeze_dim::<2>(1);
            let inv_opac: Tensor<2> = 1.0 - opacities;
            let noise_weight = inv_opac.powi_scalar(150.0).clamp(0.0, 1.0) * visible;
            let noise_weight_means =
                noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);
            let noise_m = (samples * noise_weight_means).clamp(-median_scale, median_scale);
            let noised_means = means + noise_m;
            let out = inner.slice_assign(s![.., 0..3], noised_means);
            Tensor::from_inner(out).require_grad()
        });

        let stats = TrainStepStats {
            num_visible,
            lr_mean,
            lr_rotation: self.config.lr_rotation,
            lr_scale: self.config.lr_scale,
            lr_coeffs: self.config.lr_coeffs_dc,
            lr_opac: self.config.lr_opac,
            loss: loss_inner,
        };

        (splats, stats)
    }

    pub async fn refine(&mut self, iter: u32, splats: Splats) -> (Splats, RefineStats) {
        let progress = iter as f32 / self.config.total_train_iters.max(1) as f32;
        // Refine manipulates the canonical (un-floored) params, so bake the
        // current 3D-filter floor into them first — split/clone/prune then see
        // the splat's true scales with no double-apply. A freshly recomputed
        // floor is attached at the end (below), once positions/count are known.
        let splats = splats.bake_min_scale();
        let device = splats.device();

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        let max_allowed_bounds = self.bounds.extent.max_element() * 100.0;

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let mut optim = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized");
        let alpha_mask = splats.opacities().lower_elem(MIN_OPACITY);
        let scales = splats.scales();

        // Note: we do NOT cull on a minimum scale. A genuinely flat splat
        // (a thin "pancake" representing a surface) legitimately has a tiny
        // smallest axis, so there's no correct min-scale threshold — the
        // non-finite check below still removes actually-degenerate splats.
        let scale_big = scales
            .clone()
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means() - bound_center).abs();
        let bound_mask = splat_dists
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Prune parameter that's NaN.
        fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
            t.clone().is_finite().bool_not().any_dim(1).squeeze_dim(1)
        }
        let transforms_bad = row_non_finite(&splats.transforms.val());
        let sh_bad = row_non_finite(&splats.sh_coeffs.val().flatten(1, 2));
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let non_finite_mask = transforms_bad.bool_or(sh_bad).bool_or(opac_bad);
        let num_pruned_non_finite = non_finite_mask
            .clone()
            .int()
            .sum()
            .into_scalar_async::<i32>()
            .await
            .expect("Failed to count non-finite splats") as u32;

        let prune_mask = alpha_mask
            .bool_or(scale_big)
            .bool_or(bound_mask)
            .bool_or(non_finite_mask);

        let (mut splats, refiner, pruned_count) =
            prune_points(splats, &mut optim, refiner, prune_mask).await;
        let mut split_inds = HashSet::new();

        // Always replace dead gaussians, so that the pruned budget is reused.
        if pruned_count > 0 {
            // Replacement weighting: opacity × visibility.
            let vis_f = refiner.vis_mask().float();
            let resampled_weights = splats.opacities() * vis_f.clone();
            let resampled_weights = resampled_weights
                .into_data_async()
                .await
                .expect("Failed to get weights")
                .try_into_vec::<f32>()
                .expect("Failed to read weights");
            let resampled_inds =
                multinomial_sample(&mut self.rng, &resampled_weights, pruned_count);
            split_inds.extend(resampled_inds);
        }

        // Force-split splats that are too big on screen (every refine). Rather
        // than killing them, we split them. One split keeps at least half of
        // each axis (`k >= 1/2`); a parent more than twice the cap needs later
        // refines to reach `split_at_screen_size`.
        //
        // Budget is how many new splats still fit under `max_splats`. With no
        // free slot the parent is left unchanged — there is no fallback to an
        // unbounded shrink. Spending the last slot on a parent that is still
        // over 2× the cap leaves both children oversized until a later refine
        // has room again (usually after a prune).
        let pre_oversized = split_inds.len();
        if self.config.split_at_screen_size > 0.0 {
            let oversized = refiner.above_screen_size(self.config.split_at_screen_size);
            let oversized_inds = oversized.argwhere_async().await;
            if oversized_inds.dims()[0] > 0 {
                let oversized_inds = oversized_inds
                    .squeeze_dim::<1>(1)
                    .into_data_async()
                    .await
                    .expect("Failed to get oversized indices")
                    .try_into_vec::<i32>()
                    .expect("Failed to read oversized indices");
                let mut budget = self
                    .config
                    .max_splats
                    .saturating_sub(splats.num_splats() + split_inds.len() as u32);
                for ind in oversized_inds {
                    if budget == 0 {
                        break;
                    }
                    if split_inds.insert(ind) {
                        budget -= 1;
                    }
                }
            }
        }
        let num_split_oversized = (split_inds.len() - pre_oversized) as u32;

        let pre_high_grad = split_inds.len();
        if iter >= self.config.growth_start_iter && iter < self.config.growth_stop_iter {
            let above_threshold = refiner.above_threshold(self.config.growth_grad_threshold);

            let threshold_count = above_threshold
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("Failed to get threshold") as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(pruned_count);

            // Saturating — cur_splats can exceed max_splats if the scene
            // was loaded above cap, and the u32 underflow would request
            // ~4B new splats.
            let cur_splats = splats.num_splats() + split_inds.len() as u32;
            let headroom = self.config.max_splats.saturating_sub(cur_splats);
            let grow_count = sample_high_grad.min(headroom);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                let weights = above_threshold.float() * refiner.refine_weight_norm.clone();
                let weights = weights
                    .into_data_async()
                    .await
                    .expect("Failed to get weights")
                    .try_into_vec::<f32>()
                    .expect("Failed to read weights");
                let growth_inds = multinomial_sample(&mut self.rng, &weights, grow_count);
                split_inds.extend(growth_inds);
            }
        }

        let num_split_high_grad = (split_inds.len() - pre_high_grad) as u32;
        let refine_count = split_inds.len();
        // Per-splat max on-screen extent. `refine_splats` uses it to shrink
        // oversized splats, floored so one refine cannot throw them apart.
        let screen_sizes = refiner.max_screen_size.clone();
        splats = self.refine_splats(&device, optim, splats, split_inds, screen_sizes, iter);

        // Update current bounds based on the splats.
        self.bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        device.memory_cleanup();

        // Recompute the per-splat 3D-filter floor against the new positions/
        // count and attach it — the floor is part of the splat from here until
        // the next refine. Past the freeze fraction we stop refreshing and leave
        // it baked in, so the tail settles against fixed params.
        if progress < MIN_SCALE_FREEZE_FRAC {
            // `splats` is already on the inner backend here, so `means()` is too.
            // No-op when there are no view cameras (e.g. unit tests).
            let means = splats.means();
            if let Some(f) =
                compute_min_scale(&means, &self.view_cams, self.config.min_scale_factor)
            {
                splats = splats.with_min_scale(f);
            }
        }

        let splat_count = splats.num_splats();

        (
            splats,
            RefineStats {
                num_added: refine_count as u32,
                num_split_oversized,
                num_split_high_grad,
                num_pruned: pruned_count,
                num_pruned_non_finite,
                total_splats: splat_count,
            },
        )
    }

    fn refine_splats(
        &mut self,
        device: &Device,
        mut optim: SplatOptim,
        mut splats: Splats,
        split_inds: HashSet<i32>,
        screen_sizes: Tensor<1>,
        iter: u32,
    ) -> Splats {
        let refine_count = split_inds.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect::<Vec<_>>(), [refine_count]),
                device,
            );

            let cur_transforms = splats.transforms.val().select(0, refine_inds.clone());
            let cur_means = cur_transforms.clone().slice(s![.., 0..3]);
            let cur_rots_raw = cur_transforms.clone().slice(s![.., 3..7]);
            let magnitudes = Tensor::clamp_min(
                Tensor::sum_dim(cur_rots_raw.clone().powi_scalar(2), 1).sqrt(),
                1e-32,
            );
            let cur_rots = cur_rots_raw.clone() / magnitudes;
            let cur_log_scale = cur_transforms.slice(s![.., 7..10]);
            let cur_sh_coeffs = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            let cur_scales = cur_log_scale.clone().exp();

            let cur_opac = sigmoid(cur_raw_opac);
            let inv_opac: Tensor<1> = 1.0 - cur_opac;
            // Child opacity `1 - (1-a)^(1/√2)`. p = 1/2 would preserve
            // transmittance for fully overlapping clones; the children are
            // offset, so use the higher exponent. Clamp only so `inv_sigmoid`
            // stays finite. Do not lift the result to `MIN_OPACITY`: that is
            // the prune threshold, and a parent below about 0.0055 would
            // otherwise come back as two children sitting exactly on it.
            let new_opac: Tensor<1> = 1.0 - inv_opac.powf_scalar(FRAC_1_SQRT_2);
            let new_raw_opac = inv_sigmoid(new_opac.clamp(SPLIT_OPACITY_EPS, 1.0 - MIN_OPACITY));

            // Smooth covariance-aware split. Per-axis shrink + second-moment
            // offset (one child at +offset, the other at -offset):
            //   k_i = 1 - (1 - k) · (σ_i² / σ_max²)
            //   offset_i = √(1 - k_i²) · σ_i
            // so each axis's mixture variance matches the parent. At the
            // ordinary k = 1/√2 the children sit two child standard deviations
            // apart, which is what the opacity exponent assumes, and a needle's
            // long axis grows to √2 σ. Replacing the offset with `(1-k)·σ`
            // would stop that growth but also move every ordinary split from
            // ±0.707σ to ±0.293σ, so the offset stays. Children inherit the
            // parent's rotation.
            let cur_scales_sq = cur_scales.clone().powi_scalar(2);
            let max_scale_sq = cur_scales_sq.clone().max_dim(1).clamp_min(1e-30);
            let ratio = cur_scales_sq / max_scale_sq;
            // Longest-axis shrink `k`. The ordinary split uses 1/√2. With
            // `split_at_screen_size` set, an on-screen giant shrinks harder,
            // but never below 1/2:
            // `k = clamp(split_at / screen, MIN_SPLIT_K, 1/√2)`.
            // Below 1/2 the offset √(1-k²) ≈ 1, so one refine would replace
            // the parent with two tiny children at ±σ. Those parents halve
            // on each refine until they fit. Splats within √2× of the cap
            // stay on the ordinary split (`k` was already ≥ 1/2).
            let k_per_axis: Tensor<2> = if self.config.split_at_screen_size > 0.0 {
                let k_max = screen_sizes
                    .select(0, refine_inds.clone())
                    .unsqueeze_dim(1)
                    .clamp_min(1e-6)
                    .recip()
                    .mul_scalar(self.config.split_at_screen_size)
                    .clamp(MIN_SPLIT_K, FRAC_1_SQRT_2);
                -(ratio * (-k_max + 1.0)) + 1.0
            } else {
                -(ratio * (1.0_f32 - FRAC_1_SQRT_2)) + 1.0
            };
            let offset_factor = (-k_per_axis.clone().powi_scalar(2) + 1.0)
                .clamp_min(0.0)
                .sqrt();
            let offset_local = offset_factor * cur_scales;
            let samples = quaternion_vec_multiply(cur_rots.clone(), offset_local);
            let new_log_scales = cur_log_scale + k_per_axis.log();
            let child_rots = cur_rots;

            let parent_transforms = Tensor::cat(
                vec![
                    cur_means.clone() - samples.clone(),
                    cur_rots_raw,
                    new_log_scales.clone(),
                ],
                1,
            );
            splats.transforms = splats
                .transforms
                .map(|t| t.select_assign(0, refine_inds.clone(), parent_transforms, Assign));
            splats.raw_opacities = splats
                .raw_opacities
                .map(|m| m.select_assign(0, refine_inds.clone(), new_raw_opac.clone(), Assign));

            // Child sits at parent_mean + samples (parent moves to
            // parent_mean - samples) — anti-correlated, centroid-preserving.
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let new_transforms =
                Tensor::cat(vec![cur_means + samples, child_rots, new_log_scales], 1);

            // Optimizer state lives on the inner (non-autodiff) device.
            let opt_device = device.clone().inner();
            let opt_inds = refine_inds.to_device(&opt_device);

            splats = map_splats_and_opt(
                splats,
                &mut optim,
                |x| Tensor::cat(vec![x, new_transforms], 0),
                |x| Tensor::cat(vec![x, cur_sh_coeffs], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x: Tensor<2>| {
                    let d1 = x.dims()[1];
                    let zeros = Tensor::zeros([refine_count, d1], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
                |x: Tensor<3>| {
                    let [_, d1, d2] = x.dims();
                    let zeros = Tensor::zeros([refine_count, d1, d2], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
                |x: Tensor<1>| {
                    let zeros = Tensor::zeros([refine_count], &opt_device);
                    let x = x.select_assign(0, opt_inds.clone(), zeros.clone(), Assign);
                    Tensor::cat(vec![x, zeros], 0)
                },
            );
        }

        // The periodic reset is the opt-in replacement for this subtractive
        // decay. Running both would stack two downward pressures, so decay
        // runs only while the reset is off. With the default (`0`) this block
        // is exactly the previous refine path.
        if self.config.opacity_reset_every == 0 {
            let train_t =
                (iter as f32 / self.config.total_train_iters.max(1) as f32).clamp(0.0, 1.0);
            let t_shrink_strength = 1.0 - train_t;
            let minus_opac = self.config.opac_decay * t_shrink_strength;

            // Lower opacity slowly over time.
            splats.raw_opacities = splats.raw_opacities.map(|f| {
                let new_opac = sigmoid(f) - minus_opac;
                inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
            });
        }

        self.optim = Some(optim);
        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats,
    optim: &mut SplatOptim,
    map_transforms: impl FnOnce(Tensor<2>) -> Tensor<2>,
    map_sh_coeffs: impl FnOnce(Tensor<3>) -> Tensor<3>,
    map_opac: impl FnOnce(Tensor<1>) -> Tensor<1>,

    map_opt_transforms: impl Fn(Tensor<2>) -> Tensor<2>,
    map_opt_sh_coeffs: impl Fn(Tensor<3>) -> Tensor<3>,
    map_opt_opac: impl Fn(Tensor<1>) -> Tensor<1>,
) -> Splats {
    splats.transforms = splats.transforms.map(map_transforms);
    optim.transforms.map_momentum(map_opt_transforms);
    splats.sh_coeffs = splats.sh_coeffs.map(map_sh_coeffs);
    optim.sh_coeffs.map_momentum(map_opt_sh_coeffs);
    splats.raw_opacities = splats.raw_opacities.map(map_opac);
    optim.opacities.map_momentum(map_opt_opac);
    splats
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
async fn prune_points(
    mut splats: Splats,
    optim: &mut SplatOptim,
    mut refiner: RefineRecord,
    prune: Tensor<1, Bool>,
) -> (Splats, RefineRecord, u32) {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    let prune_count = prune.dims()[0];
    if prune_count == 0 {
        return (splats, refiner, 0);
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return (splats, refiner, 0);
    }

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0] as u32;
    if new_points < start_splats {
        let valid_inds = valid_inds.squeeze_dim(1);
        let inner_valid_inds = valid_inds.clone().without_autodiff();
        splats = map_splats_and_opt(
            splats,
            optim,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
        );
        refiner = refiner.keep(inner_valid_inds);
    }
    (splats, refiner, start_splats - new_points)
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color<R: rand::Rng + ?Sized>(
    base: glam::Vec3,
    strength: f32,
    rng: &mut R,
) -> glam::Vec3 {
    if strength <= 0.0 {
        return base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    }
    use rand::RngExt as _;
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam_scaled::MomentumState;
    use crate::stats::RefineRecord;
    use brush_render::gaussian_splats::SplatRenderMode;
    use hashbrown::HashSet;
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .try_into_vec::<f32>()
            .expect("f32 tensor")
    }

    fn test_trainer(config: &TrainConfig, device: &Device) -> SplatTrainer {
        SplatTrainer::new_seeded(
            config,
            device,
            BoundingBox {
                center: glam::Vec3::ZERO,
                extent: glam::Vec3::ONE,
            },
            1,
        )
    }

    fn splats_with_opacity(device: &Device, raw: &[f32]) -> Splats {
        let n = raw.len();
        Splats::from_tensor_data(
            Tensor::zeros([n, 3], device),
            Tensor::ones([n, 4], device),
            Tensor::zeros([n, 3], device),
            Tensor::zeros([n, 1, 3], device),
            Tensor::<1>::from_floats(raw, device),
            SplatRenderMode::Default,
        )
    }

    #[test]
    fn opacity_reset_schedule_matches_refine_gate() {
        let off = TrainConfig::default().opacity_reset_every;
        assert_eq!(off, 0);
        for iter in [0, 1, 200, 3000, 28_500] {
            assert!(
                !opacity_reset_due(iter, 0.5, off),
                "default flag must never reset at iter {iter}"
            );
        }

        assert!(
            !opacity_reset_due(0, 0.0, 3000),
            "iteration 0 is not a reset"
        );
        assert!(!opacity_reset_due(2999, 0.1, 3000));
        assert!(opacity_reset_due(3000, 0.1, 3000));
        assert!(
            opacity_reset_due(3000, REFINE_STOP_PROGRESS, 3000),
            "the refine cutoff itself still resets"
        );
        assert!(
            !opacity_reset_due(3000, REFINE_STOP_PROGRESS + 0.001, 3000),
            "past the refine cutoff does not reset"
        );
        // `N` does not have to coincide with a refine step.
        assert!(opacity_reset_due(250, 0.01, 250));
        assert!(!opacity_reset_due(250, 0.01, 200));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn opacity_reset_clamps_and_zeroes_only_opacity_moments() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let opt_device = device.clone().inner();
        let config = TrainConfig {
            opacity_reset_every: 3000,
            ..TrainConfig::default()
        };

        // 0.0 → opacity 0.5 and 2.0 → ~0.88 are above the cap.
        // The cap raw value is the boundary. -5 and -6 stay below it.
        let cap_raw = opacity_reset_raw_value();
        let raw = [0.0f32, 2.0, cap_raw, -5.0, -6.0];
        let n = raw.len();
        let splats = splats_with_opacity(&device, &raw);

        let mut trainer = test_trainer(&config, &device);
        assert!(trainer.opacity_reset_due(3000, 0.5));
        assert!(!trainer.opacity_reset_due(200, 0.5));
        assert!(!trainer.opacity_reset_due(0, 0.0));

        let mut optim = SplatOptim::new(&config, splats.sh_degree(), &opt_device);
        optim.opacities.momentum = Some(MomentumState {
            moment_1: Tensor::<1>::from_floats([1.0, -2.0, 3.0, -4.0, 5.0], &opt_device),
            moment_2: Tensor::<1>::from_floats([0.5, 0.25, 0.125, 0.0625, 0.03125], &opt_device),
            time: 42,
        });
        let transform_m1 = Tensor::<2>::ones([n, 10], &opt_device);
        let transform_m2 = Tensor::<2>::ones([n, 10], &opt_device).mul_scalar(0.5);
        let transform_m1_before = to_vec(transform_m1.clone()).await;
        let transform_m2_before = to_vec(transform_m2.clone()).await;
        optim.transforms.momentum = Some(MomentumState {
            moment_1: transform_m1,
            moment_2: transform_m2,
            time: 7,
        });
        trainer.optim = Some(optim);

        let splats = trainer.reset_opacity(splats);
        let got = to_vec(splats.raw_opacities.val()).await;
        assert_eq!(got.len(), n);
        assert_eq!(
            got[0].to_bits(),
            cap_raw.to_bits(),
            "opacity 0.5 is clamped"
        );
        assert_eq!(
            got[1].to_bits(),
            cap_raw.to_bits(),
            "opacity ~0.88 is clamped"
        );
        assert_eq!(
            got[2].to_bits(),
            cap_raw.to_bits(),
            "the cap itself is unchanged"
        );
        assert_eq!(
            got[3].to_bits(),
            (-5.0f32).to_bits(),
            "below the cap is kept"
        );
        assert_eq!(
            got[4].to_bits(),
            (-6.0f32).to_bits(),
            "below the cap is kept"
        );

        let optim = trainer.optim.as_ref().expect("optim still installed");
        let opac = optim
            .opacities
            .momentum
            .as_ref()
            .expect("opacity moments exist");
        assert_eq!(opac.time, 42, "adam step counter is kept");
        for v in to_vec(opac.moment_1.clone()).await {
            assert_eq!(v.to_bits(), 0, "exp_avg zeroed");
        }
        for v in to_vec(opac.moment_2.clone()).await {
            assert_eq!(v.to_bits(), 0, "exp_avg_sq zeroed");
        }
        let transforms = optim
            .transforms
            .momentum
            .as_ref()
            .expect("transform moments");
        assert_eq!(transforms.time, 7, "other parameters keep their step");
        assert_eq!(
            to_vec(transforms.moment_1.clone()).await,
            transform_m1_before,
            "mean/scale/rotation moments are not zeroed"
        );
        assert_eq!(
            to_vec(transforms.moment_2.clone()).await,
            transform_m2_before,
            "mean/scale/rotation second moments are not zeroed"
        );

        // No optimizer yet: the clamp still happens.
        let mut bare = test_trainer(&config, &device);
        assert!(bare.optim.is_none());
        let again = bare.reset_opacity(splats_with_opacity(&device, &raw));
        let got = to_vec(again.raw_opacities.val()).await;
        assert_eq!(got[0].to_bits(), cap_raw.to_bits());
        assert_eq!(got[3].to_bits(), (-5.0f32).to_bits());
        assert!(bare.optim.is_none());
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn opacity_reset_disables_opac_decay_inside_refine() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let opt_device = device.clone().inner();
        let n = 4;
        // raw 0 is sigmoid 0.5. At iter 0 the full `opac_decay` is subtracted.
        let raw = [0.0f32; 4];

        let decay_config = TrainConfig {
            opac_decay: 0.2,
            ..TrainConfig::default()
        };
        let mut decay_trainer = test_trainer(&decay_config, &device);
        let decayed = decay_trainer.refine_splats(
            &device,
            SplatOptim::new(&decay_config, 0, &opt_device),
            splats_with_opacity(&device, &raw),
            HashSet::new(),
            Tensor::zeros([n], &device),
            0,
        );
        let decayed_raw = to_vec(decayed.raw_opacities.val()).await;
        let decayed_opac = 1.0 / (1.0 + (-decayed_raw[0]).exp());
        assert!(
            (decayed_opac - 0.3).abs() < 1e-4,
            "decay subtracts opac_decay at iter 0, got activated {decayed_opac}"
        );

        let reset_config = TrainConfig {
            opacity_reset_every: 3000,
            ..decay_config
        };
        let mut reset_trainer = test_trainer(&reset_config, &device);
        let kept = reset_trainer.refine_splats(
            &device,
            SplatOptim::new(&reset_config, 0, &opt_device),
            splats_with_opacity(&device, &raw),
            HashSet::new(),
            Tensor::zeros([n], &device),
            0,
        );
        for v in to_vec(kept.raw_opacities.val()).await {
            assert_eq!(
                v.to_bits(),
                0,
                "decay is skipped while the reset is enabled"
            );
        }
    }

    /// Small splats sit near the 3D-filter floor, so compensation `coef < 1`.
    /// Capping the raw sigmoid at 0.01 and then multiplying by `coef` drops a
    /// healthy splat under the `1/255` projection cutoff. The reset has to cap
    /// the compensated opacity instead, and must not brighten a splat that is
    /// already under the cap. The next refine must actually prune something
    /// else, or the empty-set guard would hide a total cull.
    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn opacity_reset_caps_effective_opacity_when_min_scale_is_set() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let opt_device = device.clone().inner();
        let config = TrainConfig {
            opacity_reset_every: 3000,
            // Keep the follow-up refine to a pure prune.
            split_at_screen_size: 0.0,
            growth_stop_iter: 0,
            ..TrainConfig::default()
        };

        // s = f = 0.001 on every axis → coef = (1/√2)³ ≈ 0.35355.
        // Raw sigmoid 0.5 is effective ≈ 0.177. An uncompensated cap of 0.01
        // would leave effective ≈ 0.00354, below 1/255.
        let floor = 0.001f32;
        let coef = (1.0f32 / 2.0f32.sqrt()).powi(3);
        let project_cutoff = 1.0f32 / 255.0;
        assert!((0.5 * coef - 0.1767767).abs() < 1e-5);
        assert!(
            OPACITY_RESET_MAX * coef < project_cutoff,
            "capping raw sigmoid would cull this splat after compensation"
        );

        let n = 3usize;
        let log_small = floor.ln();
        let faint_raw = (0.001f32 / (1.0 - 0.001)).ln();
        // 0: small but healthy (sigmoid 0.5). 1: large anchor (sigmoid 0.5).
        // 2: already faint; effective opacity stays under the cap and under 1/255.
        let log_scales = Tensor::<1>::from_floats(
            [
                log_small, log_small, log_small, 0.0, 0.0, 0.0, log_small, log_small, log_small,
            ],
            &device,
        )
        .reshape([n, 3]);
        let rots = Tensor::<1>::from_floats(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &device,
        )
        .reshape([n, 4]);
        let mut splats = Splats::from_tensor_data(
            Tensor::zeros([n, 3], &device),
            rots,
            log_scales,
            Tensor::zeros([n, 1, 3], &device),
            Tensor::<1>::from_floats([0.0, 0.0, faint_raw], &device),
            SplatRenderMode::Default,
        )
        .with_min_scale(Tensor::<1>::from_floats([floor, floor, floor], &device));

        let scales_before = to_vec(splats.log_scales()).await;
        let mut trainer = test_trainer(&config, &device);
        let mut optim = SplatOptim::new(&config, splats.sh_degree(), &opt_device);
        optim.opacities.momentum = Some(MomentumState {
            moment_1: Tensor::<1>::from_floats([1.0, -2.0, 3.0], &opt_device),
            moment_2: Tensor::<1>::from_floats([0.4, 0.5, 0.6], &opt_device),
            time: 42,
        });
        let transform_m1 = Tensor::<2>::ones([n, 10], &opt_device);
        let transform_m1_before = to_vec(transform_m1.clone()).await;
        optim.transforms.momentum = Some(MomentumState {
            moment_1: transform_m1,
            moment_2: Tensor::<2>::ones([n, 10], &opt_device),
            time: 7,
        });
        trainer.optim = Some(optim);

        splats = trainer.reset_opacity(splats);

        let effective = to_vec(splats.opacities()).await;
        assert!(
            (effective[0] - OPACITY_RESET_MAX).abs() < 1e-4,
            "small splat effective opacity, got {}",
            effective[0]
        );
        assert!(
            (effective[1] - OPACITY_RESET_MAX).abs() < 1e-4,
            "anchor effective opacity, got {}",
            effective[1]
        );
        assert!(
            effective[0] >= project_cutoff && effective[1] >= project_cutoff,
            "capped splats must stay visible to the projector"
        );
        assert!(
            effective[2] < project_cutoff,
            "faint splat must not be brightened, got {}",
            effective[2]
        );
        let raw_after = to_vec(splats.raw_opacities.val()).await;
        assert_eq!(
            raw_after[2].to_bits(),
            faint_raw.to_bits(),
            "faint raw opacity is left untouched"
        );
        assert_eq!(
            to_vec(splats.log_scales()).await,
            scales_before,
            "reset does not bake the scale floor"
        );
        assert!(splats.min_scale.is_some(), "floor stays attached");

        let optim = trainer.optim.as_ref().expect("optim still installed");
        let opac = optim.opacities.momentum.as_ref().expect("opacity moments");
        assert_eq!(opac.time, 42, "adam step counter is kept");
        for v in to_vec(opac.moment_1.clone()).await {
            assert_eq!(v.to_bits(), 0, "exp_avg zeroed");
        }
        for v in to_vec(opac.moment_2.clone()).await {
            assert_eq!(v.to_bits(), 0, "exp_avg_sq zeroed");
        }
        let transforms = optim
            .transforms
            .momentum
            .as_ref()
            .expect("transform moments");
        assert_eq!(transforms.time, 7);
        assert_eq!(
            to_vec(transforms.moment_1.clone()).await,
            transform_m1_before
        );

        trainer.refine_record = Some(RefineRecord::new(
            splats.num_splats(),
            &splats.device().inner(),
        ));
        let (splats, stats) = trainer.refine(100, splats).await;
        assert_eq!(
            stats.num_pruned, 1,
            "only the already-faint splat is pruned"
        );
        assert_eq!(
            splats.num_splats(),
            2,
            "the small splat survives alongside the anchor"
        );
        let survived = to_vec(splats.opacities()).await;
        assert!(survived.iter().all(|a| *a >= project_cutoff));
        let baked_scale = to_vec(splats.scales()).await;
        let expected_small = floor * 2.0f32.sqrt();
        assert!(
            (baked_scale[0] - expected_small).abs() < 1e-5,
            "surviving small splat keeps its floored scale, got {}",
            baked_scale[0]
        );
    }
}
