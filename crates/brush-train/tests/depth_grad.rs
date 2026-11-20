//! A depth-only loss must move gaussian positions and leave blending weights alone.
//!
//! Two routes otherwise carry depth error into opacity. The quotient
//! `accumulated / alpha` is differentiable in alpha, and the depth loss detaches
//! that denominator. The rasterizer would also add the depth channel into the
//! alpha dot-accumulator, which changes opacity and the gaussian footprint.
//! That term is omitted. With both routes closed, depth supervision moves
//! positions and does not change opacity, rotation, or scale.

use brush_loss::{expected_depth, inverse_depth_l1};
use brush_render::bwd::render_splats_with_mode;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, RasterizationMode, SplatRenderMode, Splats};
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::{Tensor, s};

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
    let grads = splats.bwd_validate(inverse_depth_l1(rendered, gt)).await;

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
