//! Inverse-depth L1, the depth term used by gsplat-style supervision.
//!
//! Rendered depth is an alpha-weighted sum. Dividing by accumulated alpha
//! recovers expected depth, but that denominator is detached: depth error
//! must move gaussians, not change how opaque they are.

use burn::tensor::Tensor;

/// Mean absolute error between inverse depths.
///
/// A ground-truth sample counts only when it is finite and positive (`0` marks
/// an invalid depth). A non-positive prediction contributes inverse-depth `0`
/// on those pixels, which is a large error rather than a free pass. With no
/// valid pixels the result is `0`.
pub fn inverse_depth_l1_mean(pred: &[f32], gt: &[f32]) -> f32 {
    assert_eq!(
        pred.len(),
        gt.len(),
        "inverse-depth loss needs one prediction per ground-truth sample"
    );
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for (&pred, &gt) in pred.iter().zip(gt) {
        if gt.is_finite() && gt > 0.0 {
            let disp_gt = 1.0 / gt;
            let disp_pred = if pred.is_finite() && pred > 0.0 {
                1.0 / pred
            } else {
                0.0
            };
            sum += f64::from((disp_pred - disp_gt).abs());
            count += 1.0;
        }
    }
    if count == 0.0 {
        0.0
    } else {
        (sum / count) as f32
    }
}

/// Same reduction as [`inverse_depth_l1_mean`], on `[H, W]` tensors.
pub fn inverse_depth_l1(pred_depth: Tensor<2>, gt_depth: Tensor<2>) -> Tensor<1> {
    let pred_positive = pred_depth.clone().greater_elem(0.0).float();
    let disp_pred = pred_depth.clamp_min(1e-8).recip() * pred_positive;

    let gt_valid = gt_depth.clone().greater_elem(0.0);
    let valid = gt_valid.float();
    let disp_gt = gt_depth.clamp_min(1e-8).recip();

    let abs_err = (disp_pred - disp_gt).abs() * valid.clone();
    abs_err.sum() / valid.sum().clamp_min(1.0)
}

/// `accumulated / alpha`, with alpha detached.
///
/// The rendered depth channel stores `sum_i w_i z_i`. Dividing by accumulated
/// alpha recovers expected depth. Stop-gradient on the denominator so this
/// division cannot reduce the loss by changing opacity. The rasterizer also
/// keeps the depth-channel gradient out of the blending-weight derivative.
pub fn expected_depth<const D: usize>(accumulated: Tensor<D>, alpha: Tensor<D>) -> Tensor<D> {
    accumulated / alpha.detach().clamp_min(1e-10)
}

#[cfg(test)]
mod tests {
    use super::inverse_depth_l1_mean;

    #[test]
    fn inverse_depth_l1_matches_hand_value() {
        // |1/2 - 1/4| and |1/3 - 1/2|. The zero ground-truth sample is ignored,
        // and so is the non-positive prediction paired with it.
        let loss = inverse_depth_l1_mean(&[2.0, -1.0, 3.0], &[4.0, 0.0, 2.0]);
        let expected = (0.25 + (1.0_f32 / 3.0 - 0.5).abs()) / 2.0;
        assert!((loss - expected).abs() < 1e-6, "got {loss}");
    }

    #[test]
    fn inverse_depth_l1_is_zero_without_valid_pixels() {
        assert_eq!(inverse_depth_l1_mean(&[1.0, 2.0], &[0.0, -3.0]), 0.0);
    }

    #[test]
    fn non_positive_prediction_is_inverse_depth_zero() {
        // |0 - 1/2| = 0.5 when the prediction is not a positive depth.
        let loss = inverse_depth_l1_mean(&[-1.0], &[2.0]);
        assert!((loss - 0.5).abs() < 1e-6, "got {loss}");
    }

    #[test]
    fn identical_positive_depths_have_zero_loss() {
        assert_eq!(inverse_depth_l1_mean(&[1.5, 8.0], &[1.5, 8.0]), 0.0);
    }
}
