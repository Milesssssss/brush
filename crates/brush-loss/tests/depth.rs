//! Inverse-depth loss and the detached depth normalisation, on a device.

use brush_loss::{expected_depth, inverse_depth_l1, inverse_depth_l1_mean};
use burn::tensor::Tensor;

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
