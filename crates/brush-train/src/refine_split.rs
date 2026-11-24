//! Production-path coverage for the densify split.
//!
//! The scalar checks in `split` lock the formula. These tests call
//! `SplatTrainer::refine` (and one real `step`) so a broken index remap,
//! prune, or moment clear fails here.
//!
//! Exhausted-budget policy: a free slot is one new splat under `max_splats`.
//! No free slot leaves an oversized parent unchanged (no unbounded shrink).
//! The last free slot may be spent on a parent that is still more than twice
//! the screen cap; both children then stay oversized until a later refine has
//! room again.

use std::f32::consts::FRAC_1_SQRT_2;

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, TextureMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{Splats, inverse_sigmoid, render_splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, Vec3};

use crate::adam_scaled::MomentumState;
use crate::config::TrainConfig;
use crate::split::{MIN_OPACITY, MIN_SPLIT_K};
use crate::stats::RefineRecord;
use crate::train::{SplatOptim, SplatTrainer};

const CAP: f32 = 0.5;

struct Spec {
    mean: Vec3,
    /// Unit quaternion, `wxyz`, matching `quaternion_vec_multiply`.
    rot: Quat,
    /// Linear axis scales. The longest axis is the one the shrink targets.
    scale: Vec3,
    opacity: f32,
    /// Copied onto every SH channel so a splat can be recognized after a split.
    sh: f32,
    /// Max screen extent as a fraction of the image. Above [`CAP`] the splat
    /// is force-split.
    screen: f32,
}

fn k_max_for(screen: f32) -> f32 {
    (CAP / screen.max(1e-6)).clamp(MIN_SPLIT_K, FRAC_1_SQRT_2)
}

fn axis_k(scale: f32, max_scale: f32, k_max: f32) -> f32 {
    let ratio = (scale * scale) / (max_scale * max_scale).max(1e-30);
    1.0 - (1.0 - k_max) * ratio
}

fn offset_local(scale: Vec3, k_max: f32) -> Vec3 {
    let max_scale = scale.x.max(scale.y).max(scale.z);
    let axis = |s: f32| {
        let k = axis_k(s, max_scale, k_max);
        (1.0 - k * k).max(0.0).sqrt() * s
    };
    Vec3::new(axis(scale.x), axis(scale.y), axis(scale.z))
}

fn child_scale(scale: Vec3, k_max: f32) -> Vec3 {
    let max_scale = scale.x.max(scale.y).max(scale.z);
    Vec3::new(
        scale.x * axis_k(scale.x, max_scale, k_max),
        scale.y * axis_k(scale.y, max_scale, k_max),
        scale.z * axis_k(scale.z, max_scale, k_max),
    )
}

async fn f32s<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("tensor readback")
        .try_into_vec::<f32>()
        .expect("f32 tensor")
}

fn near(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

fn near_vec(a: Vec3, b: Vec3, tol: f32, what: &str) {
    assert!(
        (a - b).length() <= tol,
        "{what}: {a:?} vs {b:?} (tol {tol})"
    );
}

fn row3(data: &[f32], i: usize) -> Vec3 {
    let s = i * 3;
    Vec3::new(data[s], data[s + 1], data[s + 2])
}

fn config(max_splats: u32) -> TrainConfig {
    TrainConfig {
        max_splats,
        opac_decay: 0.0,
        growth_start_iter: 0,
        growth_stop_iter: 0,
        growth_select_fraction: 0.0,
        split_at_screen_size: CAP,
        min_scale_factor: 0.0,
        ssim_weight: 0.0,
        lpips_loss_weight: 0.0,
        mean_noise_weight: 0.0,
        background_noise_strength: 0.0,
        total_train_iters: 1000,
        ..TrainConfig::default()
    }
}

fn build_splats(specs: &[Spec], device: &Device) -> Splats {
    let n = specs.len();
    let mut means = Vec::with_capacity(n * 3);
    let mut rots = Vec::with_capacity(n * 4);
    let mut logs = Vec::with_capacity(n * 3);
    let mut sh = Vec::with_capacity(n * 3);
    let mut opac = Vec::with_capacity(n);
    for spec in specs {
        means.extend_from_slice(&spec.mean.to_array());
        rots.extend_from_slice(&[spec.rot.w, spec.rot.x, spec.rot.y, spec.rot.z]);
        logs.extend_from_slice(&[spec.scale.x.ln(), spec.scale.y.ln(), spec.scale.z.ln()]);
        sh.extend_from_slice(&[spec.sh, spec.sh, spec.sh]);
        opac.push(inverse_sigmoid(spec.opacity));
    }
    Splats::from_raw(
        means,
        rots,
        logs,
        sh,
        opac,
        brush_render::gaussian_splats::SplatRenderMode::Default,
        device,
    )
}

fn refine_record(screens: &[f32], device: &Device) -> RefineRecord {
    let n = screens.len();
    RefineRecord {
        refine_weight_norm: Tensor::<1>::zeros([n], device),
        vis_weight: Tensor::from_data(TensorData::new(vec![1.0_f32; n], [n]), device),
        max_screen_size: Tensor::from_data(TensorData::new(screens.to_vec(), [n]), device),
    }
}

fn row_moment(n: usize, cols: usize, device: &Device) -> Tensor<2> {
    let mut data = Vec::with_capacity(n * cols);
    for row in 0..n {
        let v = row as f32 + 1.0;
        data.extend(std::iter::repeat_n(v, cols));
    }
    Tensor::from_data(TensorData::new(data, [n, cols]), device)
}

fn install_momentum(optim: &mut SplatOptim, n: usize, device: &Device) {
    optim.transforms.momentum = Some(MomentumState {
        moment_1: row_moment(n, 10, device),
        moment_2: row_moment(n, 10, device),
        time: 4,
    });
    let mut sh_m2 = Vec::with_capacity(n);
    let mut sh_m1 = Vec::with_capacity(n * 3);
    for row in 0..n {
        let v = row as f32 + 1.0;
        sh_m2.push(v);
        sh_m1.extend_from_slice(&[v, v, v]);
    }
    optim.sh_coeffs.momentum = Some(MomentumState {
        moment_1: Tensor::from_data(TensorData::new(sh_m1, [n, 1, 3]), device),
        moment_2: Tensor::from_data(TensorData::new(sh_m2, [n, 1, 1]), device),
        time: 4,
    });
    optim.opacities.momentum = Some(MomentumState {
        moment_1: row_moment(n, 1, device).squeeze_dim(1),
        moment_2: row_moment(n, 1, device).squeeze_dim(1),
        time: 4,
    });
}

struct Prepared {
    trainer: SplatTrainer,
    splats: Splats,
}

fn prepare(device: &Device, specs: &[Spec], config: &TrainConfig, with_momentum: bool) -> Prepared {
    let splats = build_splats(specs, device).valid();
    let bounds = BoundingBox::from_min_max(Vec3::splat(-20.0), Vec3::splat(20.0));
    let mut trainer = SplatTrainer::new_seeded(config, device, bounds, 7);
    let opt_device = splats.device();
    let mut optim = SplatOptim::new(config, splats.sh_degree(), &opt_device);
    if with_momentum {
        install_momentum(&mut optim, specs.len(), &opt_device);
    }
    trainer.optim = Some(optim);
    let screens: Vec<f32> = specs.iter().map(|s| s.screen).collect();
    trainer.testing_set_refine_record(refine_record(&screens, &opt_device));
    Prepared { trainer, splats }
}

async fn device() -> Device {
    burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

fn assert_finite(data: &[f32], what: &str) {
    assert!(
        data.iter().all(|v| v.is_finite()),
        "{what} has a non-finite value"
    );
}

fn assert_dims(splats: &Splats, optim: &SplatOptim) {
    let n = splats.num_splats() as usize;
    assert_eq!(splats.transforms.dims()[0], n, "transform rows");
    assert_eq!(splats.transforms.dims()[1], 10, "transform columns");
    assert_eq!(splats.sh_coeffs.dims()[0], n, "sh rows");
    assert_eq!(splats.raw_opacities.dims()[0], n, "opacity rows");
    let transforms = optim
        .transforms
        .momentum
        .as_ref()
        .expect("transform momentum");
    let sh = optim.sh_coeffs.momentum.as_ref().expect("sh momentum");
    let opac = optim.opacities.momentum.as_ref().expect("opacity momentum");
    assert_eq!(transforms.moment_1.dims()[0], n, "transform moment rows");
    assert_eq!(transforms.moment_2.dims()[0], n, "transform moment2 rows");
    assert_eq!(sh.moment_1.dims(), [n, 1, 3], "sh moment shape");
    assert_eq!(sh.moment_2.dims(), [n, 1, 1], "reduced sh moment shape");
    assert_eq!(opac.moment_1.dims(), [n], "opacity moment rows");
}

fn tiny_batch() -> SceneBatch {
    let pixel = 0xffccccccu32 as i32;
    SceneBatch {
        img_packed: TensorData::new(vec![pixel; 32 * 32], [32usize, 32]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        depth: None,
        view_index: 0,
        camera: Camera::new(
            Vec3::new(0.0, 0.0, -4.0),
            Quat::IDENTITY,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_4,
            glam::vec2(0.5, 0.5),
            Pinhole,
        ),
    }
}

async fn render_alpha(splats: &Splats, camera: &Camera, size: glam::UVec2) -> Vec<f32> {
    let (rendered, _) = render_splats(
        splats.clone(),
        camera,
        size,
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    f32s(rendered)
        .await
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| pixel[3])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn rotated() -> Quat {
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
    }

    fn fixture() -> Vec<Spec> {
        vec![
            // Just over the cap, so the shrink stays at the ordinary 1/√2.
            Spec {
                mean: Vec3::new(-0.4, 0.2, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::new(1.0, 0.25, 0.1),
                opacity: 0.6,
                sh: 0.2,
                screen: 0.6,
            },
            // Not selected. This row must survive untouched.
            Spec {
                mean: Vec3::new(1.2, -0.8, 0.15),
                rot: Quat::IDENTITY,
                scale: Vec3::new(0.3, 0.3, 0.3),
                opacity: 0.7,
                sh: 0.4,
                screen: 0.05,
            },
            // Exactly 2× the cap: k = 0.5 without needing the floor.
            Spec {
                mean: Vec3::new(0.1, 0.3, -0.2),
                rot: Quat::IDENTITY,
                scale: Vec3::new(0.4, 0.4, 0.4),
                opacity: 0.55,
                sh: 0.6,
                screen: 1.0,
            },
            // 10× cap, long axis along local X, rotated 90° about Z.
            Spec {
                mean: Vec3::new(0.3, -0.25, 0.1),
                rot: rotated(),
                scale: Vec3::new(2.0, 0.15, 0.08),
                opacity: 0.5,
                sh: 0.8,
                screen: 5.0,
            },
            // 90× cap. The floor, not 1/90, has to win.
            Spec {
                mean: Vec3::new(-1.0, 0.5, 0.05),
                rot: Quat::IDENTITY,
                scale: Vec3::new(0.7, 0.7, 0.7),
                opacity: 0.45,
                sh: 0.3,
                screen: 45.0,
            },
        ]
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn refine_splits_ordinary_twox_and_severe_on_device() {
        let device = device().await;
        let specs = fixture();
        let bystander = 1usize;
        let Prepared {
            mut trainer,
            splats,
        } = prepare(&device, &specs, &config(64), true);
        let before_tf = f32s(splats.transforms.val()).await;

        let (splats, stats) = trainer.refine(0, splats).await;

        assert_eq!(stats.num_pruned, 0, "healthy splats are not pruned");
        assert_eq!(stats.num_split_high_grad, 0, "growth is disabled");
        assert_eq!(stats.num_split_oversized, 4, "four screen-size splits");
        assert_eq!(stats.num_added, 4, "one child appended per split");
        assert_eq!(splats.num_splats(), 9, "5 parents + 4 children");

        let means = f32s(splats.means()).await;
        let scales = f32s(splats.scales()).await;
        let rots = f32s(splats.rotations()).await;
        let sh = f32s(splats.sh_coeffs.val()).await;
        let opac = f32s(splats.opacities()).await;
        let tf = f32s(splats.transforms.val()).await;
        assert_finite(&means, "means");
        assert_finite(&scales, "scales");
        assert_finite(&rots, "rotations");
        assert_finite(&sh, "sh");
        assert_finite(&opac, "opacities");

        let by_row = &tf[bystander * 10..(bystander + 1) * 10];
        let by_before = &before_tf[bystander * 10..(bystander + 1) * 10];
        for (got, exp) in by_row.iter().zip(by_before) {
            near(*got, *exp, 1e-5, "unselected transform row");
        }

        let n0 = specs.len();
        let mut claimed = vec![false; stats.num_added as usize];
        for (index, spec) in specs.iter().enumerate() {
            if spec.screen <= CAP {
                continue;
            }
            let k_max = k_max_for(spec.screen);
            let offset = spec.rot * offset_local(spec.scale, k_max);
            let expected_scale = child_scale(spec.scale, k_max);
            near_vec(
                row3(&means, index),
                spec.mean - offset,
                2e-3,
                "in-place child is the minus offset",
            );
            near_vec(
                row3(&scales, index),
                expected_scale,
                2e-3,
                "in-place child scale",
            );

            let mut partner = None;
            for (child_i, taken) in claimed.iter_mut().enumerate() {
                if *taken {
                    continue;
                }
                let got = row3(&scales, n0 + child_i);
                if (got - expected_scale).length() < 2e-2 {
                    *taken = true;
                    partner = Some(n0 + child_i);
                    break;
                }
            }
            let partner = partner.expect("appended child with the parent's new scale");
            near_vec(
                row3(&means, partner),
                spec.mean + offset,
                2e-3,
                "appended child is the plus offset",
            );
            let centroid = (row3(&means, index) + row3(&means, partner)) * 0.5;
            near_vec(
                centroid,
                spec.mean,
                2e-3,
                "split preserves the parent centroid",
            );
            let sh_parent = &sh[index * 3..index * 3 + 3];
            let sh_child = &sh[partner * 3..partner * 3 + 3];
            for (got, exp) in sh_child.iter().zip(sh_parent) {
                near(*got, *exp, 1e-5, "child inherits parent SH");
            }
            near(sh_parent[0], spec.sh, 1e-5, "parent SH is kept");
        }
        assert!(
            claimed.iter().all(|c| *c),
            "every appended child was matched to a parent"
        );

        // 90× must halve, not shrink by ~1/90.
        let severe = child_scale(specs[4].scale, k_max_for(45.0));
        near_vec(row3(&scales, 4), severe, 2e-3, "90× parent halves");
        assert!(
            (row3(&scales, 4).x / specs[4].scale.x) > 0.4,
            "90× shrink fell through the floor"
        );

        let optim = trainer.optim.as_ref().expect("optimizer kept");
        assert_dims(&splats, optim);
        assert_eq!(
            optim.transforms.momentum.as_ref().expect("moments").time,
            4,
            "a split clears moment rows but does not reset the adam step counter"
        );
        let moment = f32s(
            optim
                .transforms
                .momentum
                .as_ref()
                .expect("moments")
                .moment_1
                .clone(),
        )
        .await;
        assert_finite(&moment, "moments");
        for index in 0..specs.len() {
            let v = moment[index * 10];
            if specs[index].screen <= CAP {
                near(v, index as f32 + 1.0, 1e-5, "survivor moment");
            } else {
                near(v, 0.0, 1e-5, "split parent moment cleared");
            }
        }
        for child in n0..splats.num_splats() as usize {
            near(moment[child * 10], 0.0, 1e-5, "new child moment cleared");
        }

        let batch = tiny_batch();
        let alpha = render_alpha(&splats, &batch.camera, glam::uvec2(32, 32)).await;
        assert_finite(&alpha, "rendered alpha");
        assert!(alpha.iter().any(|&value| value > 0.05));
        let (splats, step_stats) = trainer.step(batch, splats.train()).await;
        assert!(step_stats.num_visible > 0);
        let loss = f32s(step_stats.loss).await;
        assert_finite(&loss, "training loss");
        assert!(!loss.is_empty(), "loss should be a readable scalar");
        let after = f32s(splats.transforms.val()).await;
        assert_finite(&after, "params after a training step");
        let after_sh = f32s(splats.sh_coeffs.val()).await;
        assert!(
            after_sh[n0 * 3..]
                .iter()
                .zip(&sh[n0 * 3..])
                .any(|(current, before)| (current - before).abs() > 1e-6),
            "a new child with cleared momentum must learn from the visible image"
        );
        let optim = trainer.optim.as_ref().expect("optimizer after step");
        assert_dims(&splats.valid(), optim);
        assert_eq!(
            optim
                .transforms
                .momentum
                .as_ref()
                .expect("moments after step")
                .time,
            5,
            "the training step after a split advances adam"
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn refine_splits_a_splat_measured_by_the_training_render() {
        let device = device().await;
        let config = TrainConfig {
            split_at_screen_size: 0.18,
            ..config(16)
        };
        let specs = [
            Spec {
                mean: Vec3::ZERO,
                rot: rotated(),
                scale: Vec3::new(0.85, 0.11, 0.09),
                opacity: 0.6,
                sh: 0.2,
                screen: 0.0,
            },
            Spec {
                mean: Vec3::new(0.65, 0.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.06),
                opacity: 0.6,
                sh: 0.4,
                screen: 0.0,
            },
        ];
        let bounds = BoundingBox::from_min_max(Vec3::splat(-5.0), Vec3::splat(5.0));
        let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);
        let splats = build_splats(&specs, &device).train();
        let (splats, step) = trainer.step(tiny_batch(), splats).await;
        assert_eq!(step.num_visible, 2);
        let means = f32s(splats.means()).await;
        let scales = f32s(splats.scales()).await;

        let (splats, refine) = trainer.refine(1, splats.valid()).await;
        assert_eq!(refine.num_split_oversized, 1);
        assert_eq!(refine.num_added, 1);
        assert_eq!(refine.total_splats, 3);
        let refined_means = f32s(splats.means()).await;
        let refined_scales = f32s(splats.scales()).await;
        near_vec(
            (row3(&refined_means, 0) + row3(&refined_means, 2)) * 0.5,
            row3(&means, 0),
            1e-5,
            "real split preserves the parent centroid",
        );
        for index in [0, 2] {
            near(
                row3(&refined_scales, index).x,
                row3(&scales, 0).x * 0.5,
                1e-5,
                "severe screen projection reaches the shrink floor",
            );
        }
        near_vec(
            row3(&refined_scales, 1),
            row3(&scales, 1),
            1e-5,
            "the visible smaller splat is not split",
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    #[ignore = "Manual alpha comparison of the existing offset and a more compact placement"]
    async fn compare_needle_split_offsets() {
        let device = device().await;
        let side = tiny_batch().camera;
        let end = Camera {
            position: Vec3::new(0.0, -4.0, 0.0),
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            ..side
        };
        let size = glam::uvec2(160, 160);
        for opacity in [0.2, 0.5, 0.8] {
            let specs = [Spec {
                mean: Vec3::ZERO,
                rot: rotated(),
                scale: Vec3::new(0.5, 0.035, 0.035),
                opacity,
                sh: 0.2,
                screen: 0.6,
            }];
            let Prepared {
                mut trainer,
                splats: parent,
            } = prepare(&device, &specs, &config(16), false);
            let (baseline, refine) = trainer.refine(0, parent.clone()).await;
            assert_eq!(refine.num_added, 1);

            let child_scales = f32s(baseline.scales()).await;
            let offset = specs[0].rot * (specs[0].scale - row3(&child_scales, 0));
            let mut compact = baseline.clone();
            let means =
                Tensor::from_floats([(-offset).to_array(), offset.to_array()], &compact.device());
            compact.transforms = compact
                .transforms
                .map(|values| values.slice_assign(burn::tensor::s![.., 0..3], means));

            for (view, camera) in [("side", side), ("end", end)] {
                let target = render_alpha(&parent, &camera, size).await;
                let silhouette = target.iter().filter(|&&alpha| alpha >= 0.05).count();
                assert!(silhouette > 0);
                for (variant, splats) in [("baseline", &baseline), ("compact", &compact)] {
                    let alpha = render_alpha(splats, &camera, size).await;
                    let mut outside_alpha = 0.0;
                    let mut inside_alpha = 0.0;
                    let mut inside_error = 0.0;
                    let mut parent_inside_alpha = 0.0;
                    let mut retained_pixels = 0;
                    for (&value, &reference) in alpha.iter().zip(&target) {
                        if reference >= 0.05 {
                            inside_alpha += value;
                            inside_error += (value - reference).abs();
                            parent_inside_alpha += reference;
                            retained_pixels += usize::from(value >= 0.05);
                        } else {
                            outside_alpha += value;
                        }
                    }
                    println!(
                        "opacity={opacity:.2} {view}/{variant}: outside alpha={outside_alpha:.3}, inside alpha ratio={:.3}, inside MAE={:.4}, retained silhouette={:.3}",
                        inside_alpha / parent_inside_alpha,
                        inside_error / silhouette as f32,
                        retained_pixels as f32 / silhouette as f32,
                    );
                }
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn faint_children_are_pruned_without_sitting_on_the_floor() {
        assert_faint_children_pruned(config(32)).await;
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn faint_children_are_pruned_with_default_3d_filter() {
        assert_faint_children_pruned(TrainConfig {
            min_scale_factor: TrainConfig::default().min_scale_factor,
            ..config(32)
        })
        .await;
    }

    async fn assert_faint_children_pruned(config: TrainConfig) {
        let device = device().await;
        let specs = vec![
            Spec {
                mean: Vec3::ZERO,
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.2),
                opacity: 0.005,
                sh: 0.11,
                screen: 5.0,
            },
            Spec {
                mean: Vec3::new(8.0, 0.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.25),
                opacity: 0.8,
                sh: 0.55,
                screen: 0.05,
            },
            Spec {
                mean: Vec3::new(0.0, 8.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.25),
                opacity: 0.8,
                sh: 0.88,
                screen: 0.05,
            },
        ];
        let Prepared {
            mut trainer,
            splats,
        } = prepare(&device, &specs, &config, true);
        let camera = tiny_batch().camera;
        trainer.set_view_cams(vec![(camera.position, camera.focal(glam::uvec2(32, 32)).x)]);

        let (splats, stats) = trainer.refine(0, splats).await;
        assert_eq!(splats.min_scale.is_some(), config.min_scale_factor > 0.0);
        assert_eq!(stats.num_pruned, 0, "0.005 is still above the prune floor");
        assert_eq!(stats.num_added, 1, "only the faint parent splits");
        let sh = f32s(splats.sh_coeffs.val()).await;
        // Check raw activations too, so filtering cannot hide a restored split opacity floor.
        let raw_opac = f32s(burn::tensor::activation::sigmoid(
            splats.raw_opacities.val(),
        ))
        .await;
        let opac = f32s(splats.opacities()).await;
        let mut faint = 0;
        for ((op, raw), marker) in opac.iter().zip(raw_opac).zip(sh.as_chunks::<3>().0) {
            if (marker[0] - 0.11).abs() < 1e-4 {
                faint += 1;
                assert!(
                    *op < MIN_OPACITY,
                    "faint child opacity {op} was lifted to the prune floor {MIN_OPACITY}"
                );
                assert!(
                    raw < MIN_OPACITY - 1e-4,
                    "faint child raw alpha {raw} did not fall below the prune threshold"
                );
            }
        }
        assert_eq!(faint, 2, "parent and child both carry the faint marker");

        let n = splats.num_splats() as usize;
        let opt_device = splats.device();
        trainer.testing_set_refine_record(refine_record(&vec![0.01; n], &opt_device));
        let (splats, stats) = trainer.refine(200, splats).await;
        assert_eq!(stats.num_pruned, 2, "both faint descendants are pruned");
        let sh = f32s(splats.sh_coeffs.val()).await;
        let opac = f32s(splats.opacities()).await;
        assert!(
            sh.chunks(3).all(|c| (c[0] - 0.11).abs() > 1e-3),
            "faint lineage is gone even if replacements were sampled"
        );
        assert!(
            sh.chunks(3).any(|c| (c[0] - 0.55).abs() < 1e-3),
            "healthy control 0.55 survives"
        );
        assert!(
            sh.chunks(3).any(|c| (c[0] - 0.88).abs() < 1e-3),
            "healthy control 0.88 survives"
        );
        for op in &opac {
            assert!(
                *op + 1e-5 >= MIN_OPACITY,
                "a remaining splat at {op} is still under the prune floor"
            );
        }
        assert_finite(&opac, "opacities after the prune refine");
        let optim = trainer.optim.as_ref().expect("optimizer");
        assert_dims(&splats, optim);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn exhausted_splat_budget_does_not_unbounded_shrink() {
        let device = device().await;
        // Two oversized parents and one bystander. argwhere yields the
        // oversized indices in ascending order, so a single free slot splits
        // only the first.
        let specs = vec![
            Spec {
                mean: Vec3::new(-0.5, 0.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(1.0),
                opacity: 0.6,
                sh: 0.2,
                screen: 5.0,
            },
            Spec {
                mean: Vec3::new(0.5, 0.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.8),
                opacity: 0.6,
                sh: 0.3,
                screen: 45.0,
            },
            Spec {
                mean: Vec3::new(0.0, 2.0, 0.0),
                rot: Quat::IDENTITY,
                scale: Vec3::splat(0.2),
                opacity: 0.6,
                sh: 0.4,
                screen: 0.05,
            },
        ];

        // One free slot (`count == max_splats - 1`).
        let Prepared {
            mut trainer,
            splats,
        } = prepare(&device, &specs, &config(4), true);
        let (splats, stats) = trainer.refine(0, splats).await;
        assert_eq!(stats.num_added, 1, "the last slot splits one parent");
        assert_eq!(splats.num_splats(), 4, "budget is now full");
        let scales = f32s(splats.scales()).await;
        near_vec(
            row3(&scales, 0),
            Vec3::splat(0.5),
            2e-3,
            "10× parent halves instead of shrinking to the cap",
        );
        near_vec(
            row3(&scales, 1),
            Vec3::splat(0.8),
            2e-3,
            "the second oversized parent does not get the last slot",
        );
        // Both children of the 10× parent are still far over the cap, and
        // there is no slot left, so the next refine must not keep shrinking.
        let n = splats.num_splats() as usize;
        let opt_device = splats.device();
        trainer.testing_set_refine_record(refine_record(&vec![5.0; n], &opt_device));
        let before = scales;
        let (splats, stats) = trainer.refine(200, splats).await;
        assert_eq!(stats.num_added, 0, "a full budget adds nothing");
        let after = f32s(splats.scales()).await;
        for (got, exp) in after.iter().zip(&before) {
            near(*got, *exp, 1e-4, "full budget leaves scales unchanged");
        }

        // Already at the cap: no split at all.
        let Prepared {
            mut trainer,
            splats,
        } = prepare(&device, &specs, &config(specs.len() as u32), true);
        let before = f32s(splats.scales()).await;
        let (splats, stats) = trainer.refine(0, splats).await;
        assert_eq!(stats.num_added, 0, "full budget does not split");
        assert_eq!(splats.num_splats(), specs.len() as u32, "count stays put");
        let after = f32s(splats.scales()).await;
        for (got, exp) in after.iter().zip(&before) {
            near(*got, *exp, 1e-4, "full budget does not touch scales");
        }

        // Imported over the cap (`count > max_splats`): still no new splats
        // and no unbounded emergency shrink.
        let Prepared {
            mut trainer,
            splats,
        } = prepare(&device, &specs, &config(1), true);
        let before = f32s(splats.scales()).await;
        let (splats, stats) = trainer.refine(0, splats).await;
        assert_eq!(stats.num_added, 0, "over-budget input adds nothing");
        assert_eq!(
            splats.num_splats(),
            specs.len() as u32,
            "over-budget input is not force-deleted"
        );
        let after = f32s(splats.scales()).await;
        for (got, exp) in after.iter().zip(&before) {
            near(
                *got,
                *exp,
                1e-4,
                "over-budget input is not emergency-shrunk",
            );
        }
    }
}
