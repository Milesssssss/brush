//! Depth-map math that does not touch the renderer: COLMAP scale pairs,
//! nearest resampling, and the turbo colormap used by the viewer.

use glam::{Quat, Vec3};

/// Google's turbo colormap, evaluated as the degree-6 polynomial from the
/// original reference. `t` is clamped to `[0, 1]`.
#[allow(clippy::excessive_precision)]
pub fn turbo(t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0);
    const C0: [f32; 3] = [0.1140890109226559, 0.06288340699912215, 0.2248337216805064];
    const C1: [f32; 3] = [6.716419496985708, 3.182286745507602, 7.571581586103393];
    const C2: [f32; 3] = [-66.09402360453038, -4.9279827041226, -10.09439367561635];
    const C3: [f32; 3] = [228.7660791526501, 25.04986699771574, -91.54105330182436];
    const C4: [f32; 3] = [-334.8351565777451, -69.31749712757485, 288.5858850615712];
    const C5: [f32; 3] = [218.7637218434795, 67.52150567819112, -305.2045772184957];
    const C6: [f32; 3] = [-52.88903478218835, -21.54527364654712, 110.5174647748972];
    let mut rgb = [0.0f32; 3];
    for i in 0..3 {
        let v =
            C0[i] + x * (C1[i] + x * (C2[i] + x * (C3[i] + x * (C4[i] + x * (C5[i] + x * C6[i])))));
        rgb[i] = v.clamp(0.0, 1.0);
    }
    rgb
}

pub fn turbo_u8(t: f32) -> [u8; 3] {
    let rgb = turbo(t);
    [
        (rgb[0] * 255.0).round() as u8,
        (rgb[1] * 255.0).round() as u8,
        (rgb[2] * 255.0).round() as u8,
    ]
}

/// RGBA8 image of a depth map. Non-positive and non-finite samples are
/// transparent and do not contribute to the color scale.
pub struct DepthColorImage {
    pub rgba: Vec<u8>,
    pub min: f32,
    pub max: f32,
}

pub fn colorize_depth(depth: &[f32], width: usize, height: usize) -> DepthColorImage {
    assert_eq!(
        depth.len(),
        width.saturating_mul(height),
        "depth buffer must cover every pixel"
    );
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &d in depth {
        if d.is_finite() && d > 0.0 {
            min = min.min(d);
            max = max.max(d);
        }
    }
    let has_range = min.is_finite() && max.is_finite();
    if !has_range {
        min = 0.0;
        max = 1.0;
    }
    let span = (max - min).max(1e-6);
    let mut rgba = vec![0u8; depth.len() * 4];
    for (i, &d) in depth.iter().enumerate() {
        if d.is_finite() && d > 0.0 {
            let t = ((d - min) / span).clamp(0.0, 1.0);
            let rgb = turbo_u8(t);
            let o = i * 4;
            rgba[o] = rgb[0];
            rgba[o + 1] = rgb[1];
            rgba[o + 2] = rgb[2];
            rgba[o + 3] = 255;
        }
    }
    DepthColorImage { rgba, min, max }
}

/// Nearest-neighbour resize. Identical sizes return a copy.
pub fn resample_nearest(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    assert_eq!(
        src.len(),
        src_w.saturating_mul(src_h),
        "source buffer must cover every pixel"
    );
    assert!(
        src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0,
        "resample sizes must be non-zero"
    );
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut dst = vec![0.0; dst_w * dst_h];
    for y in 0..dst_h {
        let sy = (((y as f32 + 0.5) * src_h as f32) / dst_h as f32).floor() as usize;
        let sy = sy.min(src_h - 1);
        for x in 0..dst_w {
            let sx = (((x as f32 + 0.5) * src_w as f32) / dst_w as f32).floor() as usize;
            let sx = sx.min(src_w - 1);
            dst[y * dst_w + x] = src[sy * src_w + sx];
        }
    }
    dst
}

/// Map a COLMAP landmark from reconstruction pixels into a depth map.
///
/// `xy` is in the camera's recorded width and height. A depth map at another
/// resolution is sampled at the same fractional position, rounded to the
/// nearest depth pixel. `None` when the mapped pixel is outside the map.
pub fn depth_pixel_for_landmark(
    xy: glam::Vec2,
    camera_width: u32,
    camera_height: u32,
    depth_width: usize,
    depth_height: usize,
) -> Option<(usize, usize)> {
    if camera_width == 0
        || camera_height == 0
        || depth_width == 0
        || depth_height == 0
        || !xy.x.is_finite()
        || !xy.y.is_finite()
    {
        return None;
    }
    let u = xy.x * depth_width as f32 / camera_width as f32;
    let v = xy.y * depth_height as f32 / camera_height as f32;
    if !u.is_finite() || !v.is_finite() || u < 0.0 || v < 0.0 {
        return None;
    }
    let u = u.round() as usize;
    let v = v.round() as usize;
    (u < depth_width && v < depth_height).then_some((u, v))
}

/// View-space depth of a world point under a COLMAP world-to-camera pose.
pub fn colmap_camera_depth(rotation_w2c: Quat, translation_w2c: Vec3, xyz: Vec3) -> f32 {
    (rotation_w2c * xyz + translation_w2c).z
}

/// Global scale `sum(metric) / sum(colmap)` over paired positive finite depths.
///
/// This is the scale that maps the COLMAP reconstruction into the depth maps'
/// units. `None` when no pair is usable or the result is not a positive finite
/// number.
pub fn metric_scale_from_pairs(colmap_depths: &[f32], metric_depths: &[f32]) -> Option<f32> {
    let mut colmap_sum = 0.0f64;
    let mut metric_sum = 0.0f64;
    for (&colmap, &metric) in colmap_depths.iter().zip(metric_depths) {
        if colmap.is_finite() && metric.is_finite() && colmap > 0.0 && metric > 0.0 {
            colmap_sum += f64::from(colmap);
            metric_sum += f64::from(metric);
        }
    }
    if colmap_sum <= 0.0 {
        return None;
    }
    let scale = (metric_sum / colmap_sum) as f32;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_endpoints_are_stable() {
        assert_eq!(turbo_u8(0.0), [29, 16, 57]);
        // High end of turbo is a dark red, not white.
        assert_eq!(turbo_u8(1.0), [138, 7, 15]);
    }

    #[test]
    fn colorize_ignores_invalid_samples() {
        let image = colorize_depth(&[0.0, 2.0, f32::NAN, 4.0], 4, 1);
        assert!((image.min - 2.0).abs() < 1e-6);
        assert!((image.max - 4.0).abs() < 1e-6);
        assert_eq!(&image.rgba[0..4], &[0, 0, 0, 0]);
        assert_eq!(image.rgba[7], 255);
        assert_eq!(image.rgba[15], 255);
        assert_ne!(&image.rgba[4..7], &image.rgba[12..15]);
    }

    #[test]
    fn colorize_without_valid_depth_is_transparent() {
        let image = colorize_depth(&[0.0, -1.0], 2, 1);
        assert_eq!(image.min, 0.0);
        assert_eq!(image.max, 1.0);
        assert!(image.rgba.iter().all(|b| *b == 0));
    }

    #[test]
    fn resample_is_identity_when_sizes_match() {
        let src = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(resample_nearest(&src, 2, 2, 2, 2), src);
    }

    #[test]
    fn resample_nearest_picks_the_source_pixel_under_the_center() {
        // 2x2 -> 1x1. The output center lands in the bottom-right source pixel.
        let src = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(resample_nearest(&src, 2, 2, 1, 1), vec![3.0]);
    }

    #[test]
    fn metric_scale_is_the_ratio_of_positive_sums() {
        let scale = metric_scale_from_pairs(&[1.0, 2.0, -1.0, 0.0], &[2.0, 4.0, 9.0, 8.0]);
        assert!((scale.unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn metric_scale_rejects_empty_or_non_positive_pairs() {
        assert_eq!(metric_scale_from_pairs(&[], &[]), None);
        assert_eq!(metric_scale_from_pairs(&[0.0, -2.0], &[1.0, 1.0]), None);
    }

    #[test]
    fn landmark_pixels_scale_into_the_depth_map() {
        // Same resolution: the landmark pixel is unchanged.
        assert_eq!(
            depth_pixel_for_landmark(glam::vec2(1.0, 1.0), 4, 3, 4, 3),
            Some((1, 1))
        );
        // Half resolution: camera pixel (2, 2) on an 8×6 image is depth pixel (1, 1).
        assert_eq!(
            depth_pixel_for_landmark(glam::vec2(2.0, 2.0), 8, 6, 4, 3),
            Some((1, 1))
        );
        // Indexing the half-resolution map with the raw camera pixel is outside it.
        assert_eq!(
            depth_pixel_for_landmark(glam::vec2(2.0, 2.0), 4, 3, 2, 2),
            Some((1, 1))
        );
        assert!(depth_pixel_for_landmark(glam::vec2(-1.0, 0.0), 4, 3, 4, 3).is_none());
    }

    #[test]
    fn camera_depth_is_view_space_z() {
        let depth = colmap_camera_depth(
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 10.0),
        );
        assert!((depth - 5.0).abs() < 1e-6);
    }
}
