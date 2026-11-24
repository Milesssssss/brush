use super::*;
use std::f64::consts::FRAC_PI_4;

use brush_dataset::scene::{SceneBatch, view_to_packed_data};
use brush_render::bounding_box::BoundingBox;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::kernels::camera_model::CameraModel::Pinhole;
use brush_render::sh::rgb_to_sh;
use brush_render::{AlphaMode, TextureMode, render_splats};
use burn::module::Module;
use glam::Quat;
use image::{DynamicImage, RgbImage, Rgba, RgbaImage};
use wasm_bindgen_test::wasm_bindgen_test;

use crate::checkpoint::{checkpoint_from_bytes, checkpoint_to_bytes};
use crate::config::TrainConfig;
use crate::eval::{eval_stats, eval_stats_with_appearance};
use crate::train::SplatTrainer;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const SIZE: u32 = 16;

async fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .unwrap()
        .try_into_vec()
        .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "tensor lengths");
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2.0e-5, "{a} != {b}");
    }
}

fn fixture(device: &Device) -> (Splats, Camera, TrainConfig, BoundingBox) {
    let splats = Splats::from_raw(
        vec![0.0, 0.0, 2.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![-1.2; 3],
        rgb_to_sh(Vec3::new(0.30, 0.40, 0.33)).to_array().to_vec(),
        vec![2.0],
        SplatRenderMode::Default,
        device,
    );
    let camera = Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        FRAC_PI_4,
        FRAC_PI_4,
        glam::Vec2::splat(0.5),
        Pinhole,
    );
    let config = TrainConfig {
        appearance_compensation: true,
        ssim_weight: 0.0,
        lr_mean: 1e-12,
        lr_mean_end: 1e-12,
        lr_scale: 0.0,
        lr_rotation: 0.0,
        lr_coeffs_dc: 0.0,
        lr_opac: 0.0,
        mean_noise_weight: 0.0,
        background_noise_strength: 0.0,
        ..TrainConfig::default()
    };
    let bounds = BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE);
    (splats, camera, config, bounds)
}

fn batch(
    image: DynamicImage,
    camera: Camera,
    alpha_mode: AlphaMode,
    view_index: usize,
) -> SceneBatch {
    let (img_packed, has_alpha) = view_to_packed_data(image, alpha_mode);
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode,
        view_index,
        camera,
        depth: None,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn affine_preserves_alpha_and_the_synthetic_background() {
    let device = brush_cube::test_helpers::test_device().await.into();
    let rgba = Tensor::from_floats([[[0.2, 0.4, 0.6, 0.0], [0.3, 0.4, 0.5, 0.5]]], &device);
    let delta = Tensor::from_floats(
        [
            [1.0, 0.25, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, -0.1],
        ],
        &device,
    );
    let output = correct_image(rgba, delta, Vec3::new(0.2, 0.4, 0.6));
    assert_close(
        &values(output).await,
        &[0.2, 0.4, 0.6, 0.0, 0.6, 0.45, 0.45, 0.5],
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn mean_affine_anchor_corrects_the_sampled_gradient_for_view_count() {
    let device = brush_cube::test_helpers::test_device().await.into();
    let mut appearance = Appearance::new(["a", "b", "c"].map(PathBuf::from).to_vec(), &device);
    appearance.params = Tensor::from_floats(
        [0.3, -0.1, 0.4].map(|gain| [[gain, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]]),
        &device,
    );
    let delta = appearance.for_training(0);
    let loss = appearance.regularization(&delta);
    let expected_loss = (0.09 * IDENTITY_WEIGHT + 0.04 * MEAN_IDENTITY_WEIGHT) / 12.0;
    assert_close(&values(loss.clone()).await, &[expected_loss]);
    let grad = values(delta.grad(&loss.backward()).unwrap()).await;
    let expected_gradient = (0.6 * IDENTITY_WEIGHT + 0.4 * MEAN_IDENTITY_WEIGHT) / 12.0;
    assert_close(&grad[..1], &[expected_gradient]);
    assert_close(&grad[1..], &[0.0; 11]);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn trainer_learns_an_exposure_change_and_resumes_by_image_path() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let inner = device.clone().inner();
    let (splats, camera, config, bounds) = fixture(&device);
    let (canonical, aux) = render_splats(
        splats.clone().valid(),
        &camera,
        glam::UVec2::splat(SIZE),
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    assert!(aux.num_visible > 0, "the scene fixture must render");
    let rgb: Vec<u8> = values(canonical)
        .await
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| {
            [1.45, 0.8, 1.15]
                .into_iter()
                .zip(pixel)
                .map(|(gain, rgb)| (rgb * gain * 255.0).round() as u8)
        })
        .collect();
    let exposed = DynamicImage::ImageRgb8(RgbImage::from_raw(SIZE, SIZE, rgb).unwrap());
    let frame = PathBuf::from("images/exposed.png");
    let other = PathBuf::from("more/exposed.png");
    let known = vec![frame.clone(), other.clone()];
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    trainer.set_appearance_views(known, &device);
    let train_batch = batch(exposed.clone(), camera, AlphaMode::Transparent, 0);
    let mut splats = splats;
    for _ in 0..16 {
        (splats, _) = trainer.step(train_batch.clone(), splats).await;
    }

    assert_close(
        &values(trainer.appearance_for_view(&other).unwrap()).await,
        &[0.0; 12],
    );
    assert!(
        trainer
            .appearance_for_view(Path::new("held-out/exposed.png"))
            .is_none(),
        "a held-out path with the same basename must stay outside the training-view group"
    );
    let corrected = eval_stats_with_appearance(
        splats.clone().valid(),
        &camera,
        exposed.clone(),
        AlphaMode::Transparent,
        &inner,
        trainer.appearance_for_view(&frame),
    )
    .await
    .unwrap();
    let uncorrected = eval_stats(
        splats.clone().valid(),
        &camera,
        exposed,
        AlphaMode::Transparent,
        &inner,
    )
    .await
    .unwrap();
    let corrected_psnr = corrected.psnr.into_scalar_async::<f32>().await.unwrap();
    let uncorrected_psnr = uncorrected.psnr.into_scalar_async::<f32>().await.unwrap();
    assert!(
        corrected_psnr > uncorrected_psnr + 3.0,
        "corrected PSNR {corrected_psnr}, uncorrected PSNR {uncorrected_psnr}"
    );

    let blob = checkpoint_to_bytes(&trainer, &splats, 16, &[])
        .await
        .unwrap();
    let loaded = checkpoint_from_bytes(&blob, &config, &inner, 42).unwrap();
    let mut resumed = loaded.trainer;
    resumed.set_appearance_views(vec![other.clone(), frame.clone()], &device);
    assert_close(
        &values(resumed.appearance_for_view(&frame).unwrap()).await,
        &values(trainer.appearance_for_view(&frame).unwrap()).await,
    );

    let mut resumed_batch = train_batch.clone();
    resumed_batch.view_index = 1;
    let _ = trainer.step(train_batch, splats).await;
    let _ = resumed.step(resumed_batch, loaded.splats.train()).await;
    assert_close(
        &values(resumed.appearance_for_view(&frame).unwrap()).await,
        &values(trainer.appearance_for_view(&frame).unwrap()).await,
    );

    let new_view = PathBuf::from("images/new.png");
    resumed.set_appearance_views(vec![other, frame, new_view.clone()], &device);
    assert_close(
        &values(resumed.appearance_for_view(&new_view).unwrap()).await,
        &[0.0; 12],
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn masked_background_color_does_not_train_appearance() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (splats, camera, config, bounds) = fixture(&device);
    let config = TrainConfig {
        ssim_weight: TrainConfig::default().ssim_weight,
        background_color: vec![0.2, 0.1, 0.3],
        #[cfg(not(target_family = "wasm"))]
        lpips_loss_weight: 0.5,
        ..config
    };
    let path = PathBuf::from("images/masked.png");
    let mut first = SplatTrainer::new(&config, &device, bounds);
    let mut second = SplatTrainer::new(&config, &device, bounds);
    first.set_appearance_views(vec![path.clone()], &device);
    second.set_appearance_views(vec![path.clone()], &device);
    let image = |background| {
        DynamicImage::ImageRgba8(RgbaImage::from_fn(SIZE, SIZE, |x, y| {
            if (4..12).contains(&x) && (4..12).contains(&y) {
                Rgba([130, 42, 110, if x == 4 { 128 } else { 255 }])
            } else {
                Rgba([background, background, background, 0])
            }
        }))
    };
    let (_, stats) = first
        .step(
            batch(image(0), camera, AlphaMode::Masked, 0),
            splats.clone(),
        )
        .await;
    let (_, second_stats) = second
        .step(batch(image(255), camera, AlphaMode::Masked, 0), splats)
        .await;
    assert!(stats.num_visible > 0, "the masked fixture must render");
    assert_close(&values(stats.loss).await, &values(second_stats.loss).await);
    let first_delta = values(first.appearance_for_view(&path).unwrap()).await;
    assert!(
        first_delta.iter().any(|v| v.abs() > 1e-3),
        "foreground must train appearance"
    );
    assert_close(
        &first_delta,
        &values(second.appearance_for_view(&path).unwrap()).await,
    );
    assert_close(
        &values(first.appearance.unwrap().moment_1).await,
        &values(second.appearance.unwrap().moment_1).await,
    );
}
