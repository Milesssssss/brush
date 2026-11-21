use crate::{
    Dataset,
    config::LoadDatasetConfig,
    scene::{Scene, SceneView},
};
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};

use brush_vfs::BrushVfs;
use image::ImageError;
use itertools::{Either, Itertools};
use std::{path::Path, sync::Arc};

pub mod colmap;
pub mod nerfstudio;
pub mod realitycapture;

use thiserror::Error;

pub struct DatasetLoadResult {
    pub init_splat: Option<SplatMessage>,
    pub dataset: Dataset,
    pub warnings: Vec<String>,
    /// Scale that maps the original reconstruction into the depth maps' units.
    /// Applied to the chosen initializer (points or PLY) after that choice.
    /// `None` when scale estimation was not used.
    pub metric_scale: Option<f32>,
}

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("I/O error while loading dataset: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Error decoding camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Error when decoding format: {0}")]
    InvalidFormat(String),

    #[error("Error loading splat data: {0}")]
    PlyError(#[from] DeserializeError),

    #[error("Error loading image in data: {0}")]
    ImageError(#[from] ImageError),
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error(transparent)]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud: {0}")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error(
        "Format not recognized: only colmap, nerfstudio json and RealityCapture csv are supported"
    )]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Result<DatasetLoadResult, DatasetError> {
    let mut dataset = colmap::load_dataset(vfs.clone(), load_args).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args).await;
    }

    if dataset.is_none() {
        dataset = realitycapture::read_dataset(vfs.clone(), load_args).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let result = dataset?;

    // A dataset that parsed but has no usable training views (e.g. every image
    // was missing or filtered out) would otherwise "load" and then crash on the
    // first training batch. Reject it here with a typed error instead.
    if result.dataset.train.views.is_empty() {
        return Err(FormatError::InvalidFormat(
            "dataset contains no usable training views (all images missing or filtered out)"
                .to_owned(),
        )
        .into());
    }

    // If there's an initial ply file, override the init stream with that.
    let mut ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    ply_paths.sort();

    let main_ply = ply_paths
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "init.ply"))
        .or_else(|| ply_paths.last());

    let mut init_splat = if let Some(main_ply) = main_ply {
        log::info!("Using ply {main_ply:?} as initial point cloud.");
        let reader = vfs
            .reader_at_path(main_ply)
            .await
            .map_err(DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points).await?)
    } else {
        result.init_splat
    };
    // The PLY, like the COLMAP point cloud, is in the original reconstruction
    // frame. Apply the depth-derived scale after choosing which one is used,
    // so an overriding PLY is not left behind the scaled cameras.
    if let Some(scale) = result.metric_scale
        && let Some(splat) = init_splat.as_mut()
    {
        apply_reconstruction_scale(splat, scale);
    }

    let result = DatasetLoadResult {
        init_splat,
        dataset: result.dataset,
        warnings: result.warnings,
        metric_scale: result.metric_scale,
    };
    scale_to_meters(result, load_args.units_per_meter)
}

/// Multiply an initializer's positions by `scale` and shift log-scales by `ln(scale)`.
fn apply_reconstruction_scale(splat: &mut SplatMessage, scale: f32) {
    if !(scale.is_finite() && scale > 0.0) || scale == 1.0 {
        return;
    }
    for mean in &mut splat.data.means {
        *mean *= scale;
    }
    if let Some(log_scales) = splat.data.log_scales.as_mut() {
        let shift = scale.ln();
        for log_scale in log_scales {
            *log_scale += shift;
        }
    }
}

/// Convert a dataset from its own units to metres by dividing every position
/// by `units_per_meter`. Training then runs in metres, so metric thresholds
/// mean the same thing on a millimetre scan as on a COLMAP scene; exports
/// multiply back (see `Splats::scaled`).
fn scale_to_meters(
    mut result: DatasetLoadResult,
    units_per_meter: f32,
) -> Result<DatasetLoadResult, DatasetError> {
    if !(units_per_meter.is_finite() && units_per_meter > 0.0) {
        return Err(FormatError::InvalidFormat(format!(
            "units_per_meter must be a positive number, got {units_per_meter}"
        ))
        .into());
    }
    if units_per_meter == 1.0 {
        return Ok(result);
    }
    let to_meters = 1.0 / units_per_meter;

    let scale_scene = |scene: &mut Scene| {
        let views = scene
            .views
            .iter()
            .cloned()
            .map(|mut view| {
                view.camera.position *= to_meters;
                view
            })
            .collect();
        scene.views = Arc::new(views);
    };
    scale_scene(&mut result.dataset.train);
    if let Some(eval) = result.dataset.eval.as_mut() {
        scale_scene(eval);
    }

    if let Some(init) = result.init_splat.as_mut() {
        for m in &mut init.data.means {
            *m *= to_meters;
        }
        if let Some(log_scales) = init.data.log_scales.as_mut() {
            let shift = to_meters.ln();
            for s in log_scales {
                *s += shift;
            }
        }
    }
    Ok(result)
}

/// Resolve a bare image name (as stored by colmap / `RealityCapture`, which only
/// record a filename) to a path in the VFS by brute-force suffix search. Masks
/// are skipped so an image never resolves to its own mask.
fn find_image_by_name<'a>(vfs: &'a BrushVfs, name: &str) -> Option<&'a Path> {
    vfs.files_ending_in(name)
        .filter(|p| {
            !p.iter()
                .any(|f| f == "masks" || f.eq_ignore_ascii_case("depth"))
        })
        .min()
}

/// Convert an OpenGL/Blender camera-to-world matrix (the nerfstudio
/// `transform_matrix` convention: +X right, +Y up, +Z back) into brush's
/// camera pose (+X right, +Y down, +Z forward).
fn opengl_c2w_to_pose(mut c2w: glam::Mat4) -> (glam::Vec3, glam::Quat) {
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    (translation, rotation)
}

/// Split views into (train, eval) by selecting every `eval_split_every`-th view
/// for eval. With `None`, every view is a train view.
fn split_eval_every(
    views: Vec<SceneView>,
    eval_split_every: Option<usize>,
) -> (Vec<SceneView>, Vec<SceneView>) {
    views.into_iter().enumerate().partition_map(|(i, v)| {
        if let Some(split) = eval_split_every
            && i % split == 0
        {
            Either::Right(v)
        } else {
            Either::Left(v)
        }
    })
}

/// Locate the `points3d.{txt,bin}` belonging to one COLMAP reconstruction.
pub(crate) fn find_points3d_path<'a>(
    vfs: &'a BrushVfs,
    points_dir: &'a Path,
) -> Option<(&'a Path, bool)> {
    let path = vfs
        .files_ending_in("points3d.txt")
        .chain(vfs.files_ending_in("points3d.bin"))
        .find(|p| p.parent() == Some(points_dir))?;
    let is_binary = matches!(path.extension().and_then(|e| e.to_str()), Some("bin"));
    Some((path, is_binary))
}

/// Locate a per-image depth map in a `depth` directory, matched by stem.
///
/// `images/foo/bar.png` matches `depth/foo/bar.tiff` and `depth/bar.tiff`.
pub(crate) fn find_depth_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");

    vfs.iter_files().find(|candidate| {
        let Some(stem) = candidate.file_stem() else {
            return false;
        };
        if !(stem.eq_ignore_ascii_case(search_name) || stem.eq_ignore_ascii_case(search_stem)) {
            return false;
        }
        let depth_idx = candidate
            .components()
            .position(|c| c.as_os_str().eq_ignore_ascii_case("depth"));
        depth_idx.is_some_and(|idx| {
            let candidate_components: Vec<_> = candidate.components().collect();
            let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
            let depth_dir_subpath = &candidate_components[idx + 1..candidate_components.len() - 1];
            path_dir_components.ends_with(depth_dir_subpath)
        })
    })
}

fn find_mask_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");
    let mut search_mask = search_stem.to_owned();
    search_mask.push(".mask");
    let search_mask = &search_mask;

    vfs.iter_files().find(|candidate| {
        // For the target, we don't care about its actual extension. Lets see if either the name or stem matches.
        let Some(stem) = candidate.file_stem() else {
            return false;
        };

        // We have the name of the file a la img.png, and the stem a la img.
        // We now want to accept any of img.png.*, img.*, img.mask.*.
        if stem.eq_ignore_ascii_case(search_name)
            || stem.eq_ignore_ascii_case(search_stem)
            || stem.eq_ignore_ascii_case(search_mask)
        {
            // Find "masks" directory in candidate path
            let masks_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case("masks"));

            // Check if the image directory path ends with the directory subpath after "masks/"
            // e.g., masks/foo/bar/bla.png should match images/foo/bar/bla.jpeg
            masks_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();

                // Get directory components only (excluding filename)
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let mask_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(mask_dir_subpath)
            })
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask() {
        // Basic matching with same extension
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.png")),
            Some(Path::new("masks/img.png"))
        );
        // Different extensions are ok.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.jpeg"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.jpeg")),
            Some(Path::new("masks/img.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask_formats() {
        // Test img.png.mask format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo.png"),
            PathBuf::from("masks/foo.png.mask"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo.png")),
            Some(Path::new("masks/foo.png.mask"))
        );

        // Test img.mask.png format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/bar.jpeg"),
            PathBuf::from("masks/bar.mask.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/bar.jpeg")),
            Some(Path::new("masks/bar.mask.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_nested_dirs() {
        // Nested directories must match
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar/img.png"),
            PathBuf::from("masks/foo/bar/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo/bar/img.png")),
            Some(Path::new("masks/foo/bar/img.png"))
        );
        // Should not match wrong subpath
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/baz/img.png"),
            PathBuf::from("masks/foo/img.png"),
        ]);
        assert_eq!(find_mask_path(&vfs, Path::new("images/baz/img.png")), None);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_depth_by_stem() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar.png"),
            PathBuf::from("depth/foo/bar.tiff"),
            PathBuf::from("images/other.png"),
        ]);
        assert_eq!(
            find_depth_path(&vfs, Path::new("images/foo/bar.png")),
            Some(Path::new("depth/foo/bar.tiff"))
        );
        assert_eq!(find_depth_path(&vfs, Path::new("images/other.png")), None);

        // A depth directory at the dataset root matches by stem alone.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/view.png"),
            PathBuf::from("depth/view.tiff"),
        ]);
        assert_eq!(
            find_depth_path(&vfs, Path::new("images/view.png")),
            Some(Path::new("depth/view.tiff"))
        );

        // The depth map itself is not an RGB image.
        assert!(find_image_by_name(&vfs, "view.tiff").is_none());
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_case_insensitive() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/IMG.PNG"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/IMG.PNG")),
            Some(Path::new("masks/img.png"))
        );
    }
}
