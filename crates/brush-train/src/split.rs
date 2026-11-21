//! Bounds for the densify split in `train`.
//!
//! The trainer applies these on device. The tests in this module are the
//! scalar spec (shrink, offset, opacity), and one of them checks that
//! `train` clamps with the same constants.

/// Opacity below this is pruned. Split children must be allowed to land
/// under it; lifting them up to it resurrects a dying parent.
pub(crate) const MIN_OPACITY: f32 = 1.0 / 255.0;

/// Smallest shrink of a splat's longest axis in one refine.
///
/// With the second-moment offset `√(1-k²)·σ`, `k → 0` parks two tiny
/// children at ±σ of the parent. Halving and repeating the refine keeps
/// them attached to the parent instead.
pub(crate) const MIN_SPLIT_K: f32 = 0.5;

/// Floor that keeps `inv_sigmoid` finite. Distinct from [`MIN_OPACITY`].
pub(crate) const SPLIT_OPACITY_EPS: f32 = 1e-12;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_1_SQRT_2, SQRT_2};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    /// Longest-axis shrink. `split_at_screen_size <= 0` keeps `k = 1/√2`.
    /// Otherwise `k = clamp(split_at / screen, 1/2, 1/√2)`, with screen
    /// sizes at or below `1e-6` treated as `1e-6` (the device clamp).
    fn split_k_max(screen_size: f32, split_at_screen_size: f32) -> f32 {
        if split_at_screen_size <= 0.0 {
            return FRAC_1_SQRT_2;
        }
        let screen = screen_size.max(1e-6);
        (split_at_screen_size / screen).clamp(MIN_SPLIT_K, FRAC_1_SQRT_2)
    }

    /// Per-axis shrink and second-moment offset factor. `offset * σ` is the
    /// local child placement. This is the formula in `refine_splats`.
    fn split_axis(scale: f32, max_scale: f32, k_max: f32) -> (f32, f32) {
        let max_sq = (max_scale * max_scale).max(1e-30);
        let ratio = (scale * scale) / max_sq;
        let k = 1.0 - (1.0 - k_max) * ratio;
        let offset = (1.0 - k * k).max(0.0).sqrt();
        (k, offset)
    }

    fn split_child_opacity(parent_opacity: f32) -> f32 {
        let parent = parent_opacity.clamp(0.0, 1.0);
        let child = 1.0 - (1.0 - parent).powf(FRAC_1_SQRT_2);
        child.clamp(SPLIT_OPACITY_EPS, 1.0 - MIN_OPACITY)
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn ordinary_and_mildly_oversized_splits_stay_at_one_over_sqrt_two() {
        // Within √2× of the cap the upper clamp already wins, so the floor
        // must not move these. Cap 0.5: √2× cap ≈ 0.707 screen fraction.
        approx(split_k_max(0.05, 0.5), FRAC_1_SQRT_2);
        approx(split_k_max(0.5, 0.5), FRAC_1_SQRT_2);
        approx(split_k_max(0.7, 0.5), FRAC_1_SQRT_2);
        // Disabled screen split ignores the extent.
        approx(split_k_max(10.0, 0.0), FRAC_1_SQRT_2);
        approx(split_k_max(0.0, 0.5), FRAC_1_SQRT_2);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn two_times_cap_is_unchanged_at_one_half() {
        // screen = 2 × cap → k = 0.5 already, before and after the floor.
        approx(split_k_max(1.0, 0.5), 0.5);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn far_oversized_shrink_is_floored_at_one_half() {
        // 10× the cap used to yield k = 0.1. With offset √(1-k²)≈1 that
        // places two 0.1σ children ~20 child-σ apart.
        approx(split_k_max(5.0, 0.5), MIN_SPLIT_K);
        approx(split_k_max(45.0, 0.5), MIN_SPLIT_K);
        let k = split_k_max(5.0, 0.5);
        let (k_axis, offset) = split_axis(1.0, 1.0, k);
        approx(k_axis, 0.5);
        approx(offset, (0.75_f32).sqrt());
        // Centre-to-centre in child-σ. The old k=0.1 path is ~19.9.
        let gap_child_sigma = 2.0 * offset / k_axis;
        assert!(gap_child_sigma < 4.0, "gap {gap_child_sigma}");
        assert!((gap_child_sigma - 3.4641016).abs() < 1e-4);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn ordinary_long_axis_keeps_second_moment_offset() {
        // k = 1/√2, offset = √(1-k²) = k, so the 1σ support grows to √2 σ
        // and the centres are two child standard deviations apart. That is the geometry the
        // opacity exponent assumes, including on a needle's long axis.
        // Swapping in `(1-k)·σ` would move every ordinary split from
        // ±0.707σ to ±0.293σ, so this path is left alone.
        let k = split_k_max(0.2, 0.5);
        let (k_long, offset_long) = split_axis(10.0, 10.0, k);
        let (k_short, offset_short) = split_axis(0.1, 10.0, k);
        approx(k_long, FRAC_1_SQRT_2);
        approx(offset_long, FRAC_1_SQRT_2);
        approx(k_long + offset_long, SQRT_2);
        assert!(k_short > 0.99);
        assert!(offset_short < 0.02);
        let gap_child_sigma = 2.0 * offset_long / k_long;
        approx(gap_child_sigma, 2.0);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn healthy_child_opacity_matches_the_historical_clamp() {
        for parent in [0.01_f32, 0.1, 0.5, 0.9, 0.9999] {
            let raw: f32 = 1.0 - (1.0 - parent).powf(FRAC_1_SQRT_2);
            let historical = raw.clamp(MIN_OPACITY, 1.0 - MIN_OPACITY);
            approx(split_child_opacity(parent), historical);
        }
        // a = 0.5 → about 0.387, well above the prune floor.
        let half = split_child_opacity(0.5);
        approx(half, 0.38745267);
        assert!(half > MIN_OPACITY);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn faint_parent_is_not_lifted_to_the_prune_floor() {
        // a = 0.005 used to clamp 0.00354 up to 1/255, so neither child
        // could be pruned and the pair's alpha went up.
        let child = split_child_opacity(0.005);
        assert!(
            child < MIN_OPACITY,
            "child {child} still on the prune floor"
        );
        assert!(child > SPLIT_OPACITY_EPS);
        approx(child, 0.00353813);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn train_uses_the_same_shrink_and_opacity_bounds() {
        let src = include_str!("train.rs");
        assert!(
            src.contains(".clamp(MIN_SPLIT_K, FRAC_1_SQRT_2)"),
            "refine split must floor k at MIN_SPLIT_K"
        );
        assert!(
            src.contains(".clamp(SPLIT_OPACITY_EPS, 1.0 - MIN_OPACITY)"),
            "split opacity must not clamp up to MIN_OPACITY"
        );
    }
}
