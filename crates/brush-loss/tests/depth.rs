//! Inverse-depth loss and the detached depth normalisation, on a device.

use brush_loss::{
    expected_depth, inverse_depth_l1, inverse_depth_l1_mean, relative_inverse_depth_loss,
};
use burn::tensor::{Device, Tensor};

#[tokio::test]
async fn tensor_inverse_depth_matches_the_scalar_reduction() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let pred = [2.0f32, -1.0, 3.0, 0.0];
    let gt = [4.0f32, 0.0, 2.0, 5.0];
    let pred_t = Tensor::<1>::from_floats(pred.as_slice(), &device).reshape([2, 2]);
    let gt_t = Tensor::<1>::from_floats(gt.as_slice(), &device).reshape([2, 2]);
    let got = inverse_depth_l1(pred_t, gt_t, None)
        .into_scalar_async::<f32>()
        .await
        .expect("loss readback");
    let expected = inverse_depth_l1_mean(&pred, &gt);
    assert!(
        (got - expected).abs() < 1e-5,
        "tensor {got} vs scalar {expected}"
    );
}

#[tokio::test]
async fn tensor_ignores_non_finite_targets_like_the_scalar_reference() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let check = |pred: &[f32], gt: &[f32]| {
        let pred_t = Tensor::<1>::from_floats(pred, &device).reshape([1, pred.len()]);
        let gt_t = Tensor::<1>::from_floats(gt, &device).reshape([1, gt.len()]);
        let got = inverse_depth_l1(pred_t, gt_t, None);
        (got, inverse_depth_l1_mean(pred, gt))
    };
    for (pred, gt) in [
        (&[2.0, 2.0][..], &[2.0, f32::INFINITY][..]),
        (
            &[2.0, 2.0, 2.0, 2.0][..],
            &[2.0, f32::INFINITY, f32::NAN, 0.0][..],
        ),
        (&[1.0, 1.0][..], &[f32::INFINITY, f32::NAN][..]),
        (&[2.0, 3.0][..], &[0.0, -1.0][..]),
    ] {
        let (got, expected) = check(pred, gt);
        let got = got.into_scalar_async::<f32>().await.expect("loss readback");
        assert!(
            (got - expected).abs() < 1e-5,
            "pred {pred:?} gt {gt:?}: tensor {got} vs scalar {expected}"
        );
    }
}

#[tokio::test]
async fn masked_depth_loss_has_no_value_or_gradient_outside_the_mask() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let pred = Tensor::<2>::from_floats([[2.0, 2.0]], &device).require_grad();
    let gt = Tensor::<2>::from_floats([[2.0, 4.0]], &device);
    let mask = Tensor::<2>::from_floats([[1.0, 0.0]], &device);

    let unmasked = inverse_depth_l1(pred.clone(), gt.clone(), None);
    let unmasked_value = unmasked
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("unmasked loss");
    assert!((unmasked_value - 0.125).abs() < 1e-5, "{unmasked_value}");

    let loss = inverse_depth_l1(pred.clone(), gt, Some(mask));
    let value = loss
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("masked loss");
    assert!(
        value.abs() < 1e-6,
        "masked loss should be zero, got {value}"
    );

    let grads = loss.backward();
    let grad = pred
        .grad(&grads)
        .expect("prediction should stay in the graph")
        .into_data_async()
        .await
        .expect("grad readback")
        .try_to_vec::<f32>()
        .expect("f32");
    assert!(grad[0].abs() < 1e-6, "kept pixel {grad:?}");
    assert!(
        grad[1].abs() < 1e-6,
        "excluded pixel must not receive a depth gradient, got {grad:?}"
    );
}

#[tokio::test]
async fn fractional_mask_uses_the_weight_sum_as_the_denominator() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // |1/2 - 1/4| = 0.25. Weight 0.5 makes the numerator 0.125 and the
    // weighted mean 0.25. Dividing by 1 would report 0.125 instead.
    let pred = Tensor::<2>::from_floats([[2.0]], &device).require_grad();
    let gt = Tensor::<2>::from_floats([[4.0]], &device);
    let mask = Tensor::<2>::from_floats([[0.5]], &device);
    let loss = inverse_depth_l1(pred.clone(), gt, Some(mask));
    let value = loss
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("fractional loss");
    assert!(
        (value - 0.25).abs() < 1e-5,
        "weight 0.5 should keep the weighted mean, got {value}"
    );

    let grads = loss.backward();
    let grad = pred
        .grad(&grads)
        .expect("prediction should stay in the graph")
        .into_data_async()
        .await
        .expect("grad readback")
        .try_to_vec::<f32>()
        .expect("f32");
    // d|1/p - 1/4| / dp at p = 2 is -1/p^2 = -0.25. The weight cancels.
    assert!(
        (grad[0] - (-0.25)).abs() < 1e-4,
        "fractional-mask gradient {grad:?}"
    );
}

#[tokio::test]
async fn all_zero_mask_has_zero_loss_and_zero_gradient() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let pred = Tensor::<2>::from_floats([[2.0, 3.0]], &device).require_grad();
    let gt = Tensor::<2>::from_floats([[4.0, 8.0]], &device);
    let mask = Tensor::<2>::from_floats([[0.0, 0.0]], &device);
    let loss = inverse_depth_l1(pred.clone(), gt, Some(mask));
    let value = loss
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("empty-mask loss");
    assert!(
        value.abs() < 1e-6,
        "an all-zero mask should drop the depth term, got {value}"
    );

    let grads = loss.backward();
    let grad = pred
        .grad(&grads)
        .expect("prediction should stay in the graph")
        .into_data_async()
        .await
        .expect("grad readback")
        .try_to_vec::<f32>()
        .expect("f32");
    assert!(
        grad.iter().all(|g| g.abs() < 1e-6),
        "all-zero mask gradient {grad:?}"
    );
}

#[tokio::test]
async fn expected_depth_detaches_alpha() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let accumulated = Tensor::<2>::from_floats([[2.0, 4.0]], &device).require_grad();
    let alpha = Tensor::<2>::from_floats([[0.5, 0.25]], &device).require_grad();

    let depth = expected_depth(accumulated.clone(), alpha.clone());
    let values = depth
        .clone()
        .into_data_async()
        .await
        .expect("depth readback")
        .try_to_vec::<f32>()
        .expect("f32");
    assert!((values[0] - 4.0).abs() < 1e-5);
    assert!((values[1] - 16.0).abs() < 1e-5);

    let grads = depth.sum().backward();

    let accum_grad = accumulated
        .grad(&grads)
        .expect("accumulated depth must receive a gradient")
        .into_data_async()
        .await
        .expect("accum grad")
        .try_to_vec::<f32>()
        .expect("f32");
    // d(accum / alpha) / d(accum) = 1 / alpha.
    assert!((accum_grad[0] - 2.0).abs() < 1e-4);
    assert!((accum_grad[1] - 4.0).abs() < 1e-4);

    // Alpha was detached, so it is either absent from the graph or all zeros.
    if let Some(alpha_grad) = alpha.grad(&grads) {
        let alpha_grad = alpha_grad
            .abs()
            .max()
            .into_scalar_async::<f32>()
            .await
            .expect("alpha grad");
        assert!(
            alpha_grad < 1e-8,
            "depth normalisation leaked a gradient into alpha: {alpha_grad}"
        );
    }

    // A zero alpha is clamped, not a division by zero.
    let tiny = Tensor::<2>::from_floats([[0.0]], &device);
    let safe = expected_depth(Tensor::<2>::from_floats([[1.0]], &device), tiny);
    let safe = safe
        .into_data_async()
        .await
        .expect("clamp readback")
        .try_to_vec::<f32>()
        .expect("f32");
    assert!(safe[0].is_finite());
}

async fn relative_result(
    device: &Device,
    depths: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    let image =
        |samples: &[f32]| Tensor::<1>::from_floats(samples, device).reshape([1, samples.len()]);
    let pred = image(depths).require_grad();
    let loss = relative_inverse_depth_loss(pred.clone(), image(targets), mask.map(image));
    let value = loss.clone().into_scalar_async::<f32>().await.unwrap();
    let grad = pred
        .grad(&loss.backward())
        .unwrap()
        .into_data_async()
        .await
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    (value, grad)
}

#[tokio::test]
async fn relative_depth_fits_affine_disparity_and_keeps_partial_patches() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let depth = [
        1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 2.0, 4.0,
    ];
    let matching = depth.map(|z| 4.0 / z - 1.5);
    let (matched, _) = relative_result(&device, &depth, &matching, None).await;
    assert_eq!(matched, 0.0);

    // Only the partial patch disagrees. Dropping the border makes the loss zero.
    let mut flipped_edge = matching;
    flipped_edge.swap(11, 12);
    let (loss, grad) = relative_result(&device, &depth, &flipped_edge, None).await;
    assert!(loss > 0.2);
    assert!(grad[11].abs() > 0.01 && grad[12].abs() > 0.01);

    let shifted = flipped_edge.map(|d| d * 2.0 + 10.0);
    let (transformed_loss, transformed_grad) =
        relative_result(&device, &depth, &shifted, None).await;
    assert!((loss - transformed_loss).abs() < 1e-5);
    assert!(
        grad.iter()
            .zip(transformed_grad)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
}

#[tokio::test]
async fn relative_depth_moves_a_flat_prediction_toward_target_structure() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let depth = [2.0; 3];
    let target = [3.0, 1.0, -1.0];
    let (loss, grad) = relative_result(&device, &depth, &target, None).await;
    assert!(
        loss.is_finite() && loss > 0.5,
        "flat prediction loss: {loss}"
    );
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "flat prediction gradient: {grad:?}"
    );
    assert!(
        grad[0] > 0.01 && grad[2] < -0.01,
        "wrong depth directions: {grad:?}"
    );

    let moved: Vec<_> = depth.iter().zip(&grad).map(|(z, g)| z - 0.05 * g).collect();
    let (moved_loss, _) = relative_result(&device, &moved, &target, None).await;
    assert!(
        moved_loss < loss,
        "loss after the step: {moved_loss} vs {loss}"
    );

    let scaled_depth = depth.map(|z| z * 4.0);
    let affine_target = target.map(|d| d * 2.0 + 7.0);
    let (scaled_loss, scaled_grad) =
        relative_result(&device, &scaled_depth, &affine_target, None).await;
    assert!(
        (loss - scaled_loss).abs() < 1e-5,
        "scaled loss: {scaled_loss} vs {loss}"
    );
    assert!(
        grad.iter()
            .zip(scaled_grad)
            .all(|(a, b)| (a - 4.0 * b).abs() < 1e-5),
        "changing depth units must scale the gradient inversely"
    );
}

#[tokio::test]
async fn relative_depth_mask_controls_statistics_and_empty_loss() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let depth = [1.0, 2.0, 4.0, 3.0, 1.0];
    let targets = [-2.0, -1.0, 4.0, 2.0, f32::INFINITY];
    let mask = [1.0, 0.5, 1.0, 0.0, 1.0];
    let (loss, grad) = relative_result(&device, &depth, &targets, Some(&mask)).await;
    assert!(loss > 0.2 && grad[0].abs() > 0.01);
    assert_eq!((grad[3], grad[4]), (0.0, 0.0));

    let changed = [-2.0, -1.0, 4.0, f32::MAX, f32::NAN];
    let (changed_loss, changed_grad) =
        relative_result(&device, &depth, &changed, Some(&mask)).await;
    assert_eq!(loss, changed_loss);
    assert_eq!(grad, changed_grad);

    let zeros = [0.0; 5];
    let (loss, grad) = relative_result(&device, &depth, &changed, Some(&zeros)).await;
    assert_eq!(loss, 0.0);
    assert_eq!(grad, zeros);

    let fractional = [0.25, 0.39, 1.0, 0.0, 0.16];
    let (loss, grad) = relative_result(&device, &depth, &[0.2; 5], Some(&fractional)).await;
    assert_eq!(loss, 0.0);
    assert_eq!(grad, zeros);
}
