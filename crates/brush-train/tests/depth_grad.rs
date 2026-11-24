//! A depth-only loss must move gaussian positions and leave blending weights alone.
//!
//! Two routes otherwise carry depth error into opacity. The quotient
//! `accumulated / alpha` is differentiable in alpha, and the depth loss detaches
//! that denominator. The rasterizer would also add the depth channel into the
//! alpha dot-accumulator, which changes opacity and the gaussian footprint.
//! That term is omitted. With both routes closed, depth supervision moves
//! positions and does not change opacity, rotation, or scale.

use brush_dataset::{
    config::DepthMode,
    scene::{DepthBatch, SceneBatch},
};
use brush_loss::{expected_depth, inverse_depth_l1};
use brush_render::AlphaMode;
use brush_render::bounding_box::BoundingBox;
use brush_render::bwd::render_splats_with_mode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, RasterizationMode, SplatRenderMode, Splats};
use brush_render::kernels::camera_model::CameraModel;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::tensor::{Device, Tensor, TensorData, s};
use glam::{Quat, Vec2, Vec3};

const SIDE: usize = 24;
const PIXELS: usize = SIDE * SIDE;
const CENTER: usize = SIDE * (SIDE / 2) + SIDE / 2;

fn rotated_camera() -> Camera {
    Camera::new(
        Vec3::new(0.2, -0.1, 0.4),
        Quat::from_rotation_y(0.4),
        0.8,
        0.8,
        Vec2::splat(0.5),
        CameraModel::Pinhole,
    )
}

fn splats_at(camera: Camera, points: &[Vec3], device: &Device) -> Splats {
    let n = points.len();
    Splats::from_raw(
        points
            .iter()
            .flat_map(|&p| (camera.position + camera.rotation * p).to_array())
            .collect(),
        (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
        vec![-1.5; n * 3],
        vec![-3.0; n * 3],
        vec![2.0; n],
        SplatRenderMode::Default,
        device,
    )
}

fn batch(
    camera: Camera,
    depth: Option<(DepthMode, Vec<f32>)>,
    mask: Option<Vec<u8>>,
) -> SceneBatch {
    let has_alpha = mask.is_some();
    SceneBatch {
        img_packed: TensorData::new(
            mask.unwrap_or_else(|| vec![255; PIXELS])
                .into_iter()
                .map(|alpha| i32::from_le_bytes([0, 0, 0, alpha]))
                .collect::<Vec<_>>(),
            [SIDE, SIDE],
        ),
        has_alpha,
        alpha_mode: if has_alpha {
            AlphaMode::Masked
        } else {
            AlphaMode::Transparent
        },
        depth: depth.map(|(mode, data)| DepthBatch {
            mode,
            data: TensorData::new(data, [SIDE, SIDE]),
        }),
        camera,
    }
}

fn trainer(device: &Device, depth_loss_weight: f32) -> SplatTrainer {
    SplatTrainer::new(
        &TrainConfig {
            depth_loss_weight,
            ssim_weight: 0.0,
            lr_mean: 0.01,
            lr_mean_end: 0.01,
            mean_noise_weight: 0.0,
            background_noise_strength: 0.0,
            ..TrainConfig::default()
        },
        device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    )
}

async fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec()
        .unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "tensor readbacks differ in length");
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance, "index {i}: {a} vs {b}");
    }
}

#[tokio::test]
async fn depth_loss_moves_positions_not_opacity() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

    let means = vec![
        0.0, 0.0, 0.0, //
        0.3, 0.0, 0.5, //
        -0.3, 0.2, 1.0, //
        0.1, -0.2, 1.5, //
    ];
    let n = means.len() / 3;
    let rotations: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..n).flat_map(|_| [-1.0, -1.0, -1.0]).collect();
    let sh: Vec<f32> = (0..n).flat_map(|_| [0.5, 0.5, 0.5]).collect();
    let opac = vec![4.0; n];

    let splats = Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh,
        opac,
        SplatRenderMode::Default,
        &device,
    );

    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.7,
        0.7,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let img_size = glam::uvec2(48, 48);
    let diff_out = render_splats_with_mode(
        splats.clone(),
        &camera,
        img_size,
        glam::Vec3::ZERO,
        RasterPass::Backward,
        RasterizationMode::RgbaAndDepth,
    )
    .await;

    let [img_h, img_w, channels] = diff_out.img.dims();
    assert_eq!(
        channels, 5,
        "depth renders append the accumulated-depth channel"
    );
    let accumulated = diff_out.img.clone().slice(s![.., .., 4..5]);
    let alpha = diff_out.img.clone().slice(s![.., .., 3..4]);
    let rendered = expected_depth(accumulated, alpha).reshape([img_h, img_w]);
    let gt = Tensor::<2>::ones([img_h, img_w], &device) * 3.0;
    let grads = splats
        .bwd_validate(inverse_depth_l1(rendered, gt, None))
        .await;

    let transforms_grad = splats
        .transforms
        .grad(&grads)
        .expect("depth loss must reach gaussian positions");
    let means_grad = transforms_grad
        .clone()
        .slice(s![.., 0..3])
        .abs()
        .max()
        .into_scalar_async::<f32>()
        .await
        .expect("position grad");
    assert!(
        means_grad > 1e-8,
        "expected a nonzero position gradient, got {means_grad}"
    );

    // Rotation and scale set the blending footprint. Depth must not use them
    // to reduce the loss.
    let pose_grad = transforms_grad
        .slice(s![.., 3..10])
        .abs()
        .max()
        .into_scalar_async::<f32>()
        .await
        .expect("rotation and scale grad");
    assert!(
        pose_grad < 1e-6,
        "depth loss leaked into rotation or scale: {pose_grad}"
    );

    if let Some(opac_grad) = splats.raw_opacities.grad(&grads) {
        let opac_grad = opac_grad
            .abs()
            .max()
            .into_scalar_async::<f32>()
            .await
            .expect("opacity grad");
        assert!(
            opac_grad < 1e-8,
            "depth loss leaked into opacity: {opac_grad}"
        );
    }
}

#[tokio::test]
async fn rotated_camera_depth_has_expected_direction_and_trainer_reduces_error() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = rotated_camera();
    let splats = splats_at(camera, &[Vec3::Z * 3.0], &device);
    let size = glam::UVec2::splat(SIDE as u32);
    let diff = render_splats_with_mode(
        splats.clone(),
        &camera,
        size,
        Vec3::ZERO,
        RasterPass::Backward,
        RasterizationMode::RgbaAndDepth,
    )
    .await;
    let [h, w, channels] = diff.img.dims();
    assert_eq!(channels, 5);
    let color = render_splats_with_mode(
        splats.clone(),
        &camera,
        size,
        Vec3::ZERO,
        RasterPass::Backward,
        RasterizationMode::Rgba,
    )
    .await;
    let rgba = values(diff.img.clone().slice(s![.., .., 0..4])).await;
    assert!(
        rgba[CENTER * 4 + 3] > 0.1,
        "fixture must cover the target pixel"
    );
    assert_close(&rgba, &values(color.img).await, 1e-6);

    let mut depth = vec![0.0; PIXELS];
    depth[CENTER] = 2.0;
    let rendered = expected_depth(
        diff.img.clone().slice(s![.., .., 4..5]),
        diff.img.slice(s![.., .., 3..4]),
    )
    .reshape([h, w]);
    let target = Tensor::from_data(TensorData::new(depth.clone(), [h, w]), &device);
    let loss = inverse_depth_l1(rendered, target, None);
    let loss_value = loss.clone().into_scalar_async::<f32>().await.unwrap();
    assert!((loss_value - (0.5 - 1.0 / 3.0)).abs() < 1e-5);
    let grads = splats.bwd_validate(loss).await;
    let actual = values(splats.transforms.grad(&grads).unwrap()).await;
    let forward = camera.rotation * Vec3::Z;
    assert_close(&actual[..3], &(forward / 9.0).to_array(), 1e-5);
    assert_close(&actual[3..], &[0.0; 7], 1e-6);

    let mut trainer = trainer(&device, 1.0);
    let batch = batch(camera, Some((DepthMode::Metric, depth)), None);
    let (after, before_stats) = trainer.step(batch.clone(), splats).await;
    let (_, after_stats) = trainer.step(batch, after).await;
    assert_eq!(before_stats.num_visible, 1);
    let before_loss = before_stats.loss.into_scalar_async::<f32>().await.unwrap();
    let after_loss = after_stats.loss.into_scalar_async::<f32>().await.unwrap();
    assert!((before_loss - loss_value).abs() < 1e-5);
    assert!(
        after_loss < before_loss - 1e-4,
        "{before_loss} vs {after_loss}"
    );
}

#[tokio::test]
async fn trainer_masks_depth_and_ignores_invalid_targets_in_both_modes() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = rotated_camera();
    let points = [
        Vec3::new(-0.4, 0.0, 2.0),
        Vec3::new(0.0, 0.0, 2.7),
        Vec3::new(0.7, 0.0, 3.5),
    ];
    let splats = splats_at(camera, &points, &device);
    let mut mask = vec![255; PIXELS];
    for row in mask.as_chunks_mut::<SIDE>().0 {
        row[17..].fill(0);
    }
    for mode in [DepthMode::Metric, DepthMode::RelativeInverse] {
        let mut target = vec![0.0; PIXELS];
        for (i, value) in target.iter_mut().enumerate() {
            *value = 1.0 + (i % SIDE) as f32 * 0.1;
        }
        target[CENTER] = f32::INFINITY;
        let mut outside_changed = target.clone();
        for (sample, &alpha) in outside_changed.iter_mut().zip(&mask) {
            if alpha == 0 {
                *sample = 9.0;
            }
        }
        outside_changed[CENTER] = f32::NAN;
        let (a, a_stats) = trainer(&device, 1.0)
            .step(
                batch(camera, Some((mode, target)), Some(mask.clone())),
                splats.clone(),
            )
            .await;
        let (b, b_stats) = trainer(&device, 1.0)
            .step(
                batch(camera, Some((mode, outside_changed)), Some(mask.clone())),
                splats.clone(),
            )
            .await;
        assert_eq!(a_stats.num_visible, 3);
        let loss_a = a_stats.loss.into_scalar_async::<f32>().await.unwrap();
        let loss_b = b_stats.loss.into_scalar_async::<f32>().await.unwrap();
        assert!(
            loss_a > 1e-4 && (loss_a - loss_b).abs() < 1e-6,
            "{mode:?}: {loss_a} vs {loss_b}"
        );
        let means_a = values(a.means()).await;
        assert_close(&means_a, &values(b.means()).await, 1e-6);
        let original_means = values(splats.means()).await;
        assert!(
            means_a
                .iter()
                .zip(&original_means)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );

        let (all_masked, stats) = trainer(&device, 1.0)
            .step(
                batch(
                    camera,
                    Some((mode, vec![5.0; PIXELS])),
                    Some(vec![0; PIXELS]),
                ),
                splats.clone(),
            )
            .await;
        assert_eq!(stats.loss.into_scalar_async::<f32>().await.unwrap(), 0.0);
        assert_close(&values(all_masked.means()).await, &original_means, 1e-7);
    }
}

#[tokio::test]
async fn zero_depth_weight_preserves_color_loss_and_optimizer_update() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let camera = rotated_camera();
    let mut splats = splats_at(camera, &[Vec3::Z * 3.0], &device);
    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|tensor| Tensor::zeros(tensor.dims(), &device).require_grad());
    let mut baseline = batch(camera, None, None);
    baseline.img_packed = TensorData::new(
        vec![i32::from_le_bytes([64, 128, 192, 255]); PIXELS],
        [SIDE, SIDE],
    );
    let mut with_depth = baseline.clone();
    with_depth.depth = Some(DepthBatch {
        mode: DepthMode::Metric,
        data: TensorData::new(vec![1.0_f32; PIXELS], [SIDE, SIDE]),
    });
    let (baseline, a) = trainer(&device, 0.0).step(baseline, splats.clone()).await;
    let (with_depth, b) = trainer(&device, 0.0).step(with_depth, splats).await;
    let loss_a = a.loss.into_scalar_async::<f32>().await.unwrap();
    let loss_b = b.loss.into_scalar_async::<f32>().await.unwrap();
    assert!(loss_a > 0.1 && (loss_a - loss_b).abs() < 1e-6);
    assert_close(
        &values(baseline.transforms.val()).await,
        &values(with_depth.transforms.val()).await,
        1e-6,
    );
    let baseline_sh = values(baseline.sh_coeffs.val()).await;
    assert!(baseline_sh.iter().any(|&v| v.abs() > 1e-4));
    assert_close(
        &baseline_sh,
        &values(with_depth.sh_coeffs.val()).await,
        1e-6,
    );
    assert_close(
        &values(baseline.raw_opacities.val()).await,
        &values(with_depth.raw_opacities.val()).await,
        1e-6,
    );
}
