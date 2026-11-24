//! Inverse-depth L1, the depth term used by gsplat-style supervision.
//!
//! Rendered depth is an alpha-weighted sum. Dividing by accumulated alpha
//! recovers expected depth, but that denominator is detached: depth error
//! must move gaussians, not change how opaque they are.

use burn::tensor::{Bool, Tensor, ops::PadMode};

const PATCH_SIZE: usize = 11;
const DEPTH_MARGIN: f32 = 0.2;

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
///
/// `mask`, when set, is a per-pixel weight in `[0, 1]` (the ground-truth alpha
/// in masked mode). A weight of `0` drops that pixel. The mean is the weighted
/// absolute error divided by the sum of those weights, including a sum between
/// zero and one. Only an empty sum is replaced, and that case returns `0`.
pub fn inverse_depth_l1(
    pred_depth: Tensor<2>,
    gt_depth: Tensor<2>,
    mask: Option<Tensor<2>>,
) -> Tensor<1> {
    let pred_ok = positive_finite(pred_depth.clone());
    let gt_ok = positive_finite(gt_depth.clone());
    // Replace invalid samples before the reciprocal so infinity and NaN never
    // enter the subtraction. Invalid predictions then contribute disparity 0.
    let disp_pred = pred_depth
        .mask_fill(pred_ok.clone().bool_not(), 1.0)
        .recip()
        * pred_ok.float();
    let disp_gt = gt_depth.mask_fill(gt_ok.clone().bool_not(), 1.0).recip();

    let mut weight = gt_ok.float();
    if let Some(mask) = mask {
        let mask_finite = mask
            .clone()
            .greater_equal_elem(0.0)
            .bool_and(mask.clone().lower_equal_elem(1.0));
        let safe_mask = mask.mask_fill(mask_finite.bool_not(), 0.0).clamp(0.0, 1.0);
        weight = weight * safe_mask;
    }

    let abs_err = (disp_pred - disp_gt).abs() * weight.clone();
    let total = weight.sum();
    // Keep every positive weight sum, including one below 1. An empty sum
    // would divide by zero; the numerator is already 0, so any stand-in works.
    let empty = total.clone().lower_equal_elem(0.0);
    abs_err.sum() / total.mask_fill(empty, 1.0)
}

/// Finite and strictly positive. `+Inf` fails the upper bound; `NaN` fails both.
fn positive_finite(values: Tensor<2>) -> Tensor<2, Bool> {
    values
        .clone()
        .greater_elem(0.0)
        .bool_and(values.lower_equal_elem(f32::MAX))
}

/// Local and image-normalized supervision for unitless, near-high inverse depth.
/// The same weights define the statistics and the loss, so masked samples
/// cannot influence other pixels through normalization.
pub fn relative_inverse_depth_loss(
    pred_depth: Tensor<2>,
    target_inverse_depth: Tensor<2>,
    mask: Option<Tensor<2>>,
) -> Tensor<1> {
    let finite_target = target_inverse_depth
        .clone()
        .greater_equal_elem(-f32::MAX)
        .bool_and(target_inverse_depth.clone().lower_equal_elem(f32::MAX));
    let valid = positive_finite(pred_depth.clone()).bool_and(finite_target);
    let mut weights = valid.float();
    if let Some(mask) = mask {
        weights = weights * mask;
    }
    let excluded = weights.clone().equal_elem(0.0);
    let pred = pred_depth.mask_fill(excluded.clone(), 1.0).recip();
    let target = target_inverse_depth.mask_fill(excluded, 0.0);

    let weight_patches = depth_patches(weights);
    let image_weights = weight_patches.clone().reshape([1, -1]);
    let (pred_local, pred_global, _) = normalized_depth(pred, &weight_patches, &image_weights);
    let (target_local, target_global, target_std) =
        normalized_depth(target, &weight_patches, &image_weights);
    let weights = weight_patches * target_std.greater_elem(0.0).float();

    0.1 * margin_l2(pred_local - target_local, weights.clone())
        + margin_l2(pred_global - target_global, weights)
}

fn depth_patches(values: Tensor<2>) -> Tensor<2> {
    let [height, width] = values.dims();
    let patch_rows = height.div_ceil(PATCH_SIZE);
    let patch_cols = width.div_ceil(PATCH_SIZE);
    values
        .pad(
            [
                (0, patch_rows * PATCH_SIZE - height),
                (0, patch_cols * PATCH_SIZE - width),
            ],
            PadMode::Constant(0.0),
        )
        .reshape([patch_rows, PATCH_SIZE, patch_cols, PATCH_SIZE])
        .permute([0, 2, 1, 3])
        .reshape([patch_rows * patch_cols, PATCH_SIZE * PATCH_SIZE])
}

fn normalized_depth(
    values: Tensor<2>,
    weight_patches: &Tensor<2>,
    image_weights: &Tensor<2>,
) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let patches = depth_patches(values);
    let (_, image_std, pivot) =
        weighted_depth_stats(patches.clone().reshape([1, -1]), image_weights);
    let (centered, patch_std, _) = weighted_depth_stats(patches, weight_patches);
    // A flat prediction still needs a gradient. Use the magnitude of its
    // detached disparity so the gradient scales with changes in depth units.
    let fallback = image_std.clone().equal_elem(0.0).float() * nonzero(pivot.abs());
    let local_std = patch_std + image_std.clone() * 0.01 + fallback.clone();
    let global_std = image_std.clone().detach() * 1.01 + fallback;
    (
        centered.clone() / nonzero(local_std),
        centered / nonzero(global_std),
        image_std,
    )
}

fn weighted_depth_stats(
    values: Tensor<2>,
    weights: &Tensor<2>,
) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    // Center relative to a covered sample first. A flat patch then has exact
    // zero variance even with fractional weights and a rounded weighted mean.
    let pivot = values.clone().gather(1, weights.clone().argmax(1)).detach();
    let values = values - pivot.clone();
    let count = nonzero(weights.clone().sum_dim(1));
    let mean = (values.clone() * weights.clone()).sum_dim(1) / count.clone();
    let centered = (values - mean).mask_fill(weights.clone().equal_elem(0.0), 0.0);
    let var = (centered.clone().powi_scalar(2) * weights.clone()).sum_dim(1) / count;
    // A zero variance has zero derivative here; sqrt's derivative is singular
    // at zero even if a later normalization denominator is regularized.
    let zero = var.clone().equal_elem(0.0);
    let std = var.mask_fill(zero.clone(), 1.0).sqrt().mask_fill(zero, 0.0);
    (centered, std, pivot)
}

fn margin_l2(error: Tensor<2>, weights: Tensor<2>) -> Tensor<1> {
    let weights = weights.mask_fill(error.clone().abs().lower_equal_elem(DEPTH_MARGIN), 0.0);
    (error.powi_scalar(2) * weights.clone()).sum() / nonzero(weights.sum())
}

fn nonzero<const D: usize>(value: Tensor<D>) -> Tensor<D> {
    let zero = value.clone().equal_elem(0.0);
    value.mask_fill(zero, 1.0)
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

    #[test]
    fn non_finite_targets_are_ignored() {
        assert_eq!(
            inverse_depth_l1_mean(&[2.0, 2.0, 2.0, 2.0], &[2.0, f32::INFINITY, f32::NAN, 0.0]),
            0.0
        );
        assert_eq!(
            inverse_depth_l1_mean(&[1.0, 1.0], &[f32::INFINITY, f32::NAN]),
            0.0
        );
    }
}
