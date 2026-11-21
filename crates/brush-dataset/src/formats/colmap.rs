use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::{DatasetLoadResult, FormatError};
use crate::{
    Dataset,
    config::LoadDatasetConfig,
    depth::{colmap_camera_depth, depth_pixel_for_landmark, metric_scale_from_pairs},
    formats::{
        find_depth_path, find_image_by_name, find_mask_path, find_points3d_path, split_eval_every,
    },
    load_depth::LoadDepth,
    scene::{LoadImage, SceneView},
};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8, ThinPrismFisheye,
};
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_render::kernels::camera_model::thin_prism_fisheye::ThinPrismFisheyeParams;
use brush_render::{
    camera::{self, Camera},
    sh::rgb_to_sh,
};
use brush_serde::{ParseMetadata, SplatData, SplatMessage};
use brush_vfs::BrushVfs;
use colmap_reader::{ColmapCamera, ColmapCameraModel};

/// COLMAP can emit several independent sparse reconstructions (`sparse/0`,
/// `sparse/1`, ...) when the image graph is disconnected. They share no
/// coordinate frame and cannot be merged here, so we pick the one that
/// registered the most images (COLMAP's own "largest first" convention,
/// determined empirically rather than trusting directory names).
async fn select_colmap_model(vfs: &BrushVfs) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = vfs
        .files_ending_in("cameras.bin")
        .chain(vfs.files_ending_in("cameras.txt"))
        .map(Path::to_path_buf)
        .collect();

    if candidates.len() <= 1 {
        return candidates.into_iter().next();
    }

    let mut best: Option<(usize, PathBuf)> = None;
    for cam in &candidates {
        let dir = cam
            .parent()
            .expect("colmap cameras file must have a parent");
        let is_binary = cam.extension().and_then(|e| e.to_str()) == Some("bin");
        let img_path = dir.join(if is_binary {
            "images.bin"
        } else {
            "images.txt"
        });

        let Some(count) = count_registered_images(vfs, &img_path, is_binary).await else {
            log::warn!(
                "Skipping colmap model '{}': can't read images",
                dir.display()
            );
            continue;
        };
        log::info!("Colmap model '{}' registered {count} images", dir.display());

        // Tie-break on path so the choice is deterministic (VFS iteration isn't).
        let better = best
            .as_ref()
            .is_none_or(|(bc, bp)| count > *bc || (count == *bc && cam < bp));
        if better {
            best = Some((count, cam.clone()));
        }
    }

    // If every candidate failed to read, fall through to a deterministic pick
    // so the caller still surfaces a proper parse error downstream.
    let chosen = best
        .map(|(_, p)| p)
        .or_else(|| candidates.iter().min().cloned())?;
    log::info!(
        "Selected colmap model '{}'",
        chosen
            .parent()
            .expect("colmap cameras file must have a parent")
            .display()
    );
    Some(chosen)
}

async fn count_registered_images(
    vfs: &BrushVfs,
    img_path: &Path,
    is_binary: bool,
) -> Option<usize> {
    let mut file = vfs.reader_at_path(img_path).await.ok()?;
    let imgs = colmap_reader::read_images(&mut file, is_binary, false)
        .await
        .ok()?;
    Some(imgs.len())
}

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    log::info!("Loading colmap dataset");

    let cam_path = select_colmap_model(&vfs).await?;
    let dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent");
    let is_binary = cam_path.extension().and_then(|e| e.to_str()) == Some("bin");
    let img_path = dir.join(if is_binary {
        "images.bin"
    } else {
        "images.txt"
    });

    Some(load_dataset_inner(vfs, load_args, cam_path, img_path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    cam_path: PathBuf,
    img_path: PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    let is_binary = cam_path.ends_with("cameras.bin");

    log::info!("Parsing colmap camera info");

    let load_args = load_args.clone();
    let vfs = vfs.clone();

    let vfs_init = vfs.clone();

    // Resolve points3d from the same reconstruction as the chosen cameras,
    // not an arbitrary one elsewhere in the VFS.
    let points_dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent")
        .to_path_buf();
    let points_dir_for_scale = points_dir.clone();

    // One actor for both halves of the colmap load — the camera/image
    // parse and the points3d parse run concurrently on the same thread
    // (no cross-stream GPU concerns; this is pure CPU/I/O).
    let actor = brush_async::Actor::new("colmap-loader");
    let dataset = actor.run(move || async move {
        let mut cam_file = vfs.reader_at_path(&cam_path).await?;
        let cam_model_data = colmap_reader::read_cameras(&mut cam_file, is_binary).await?;
        let cam_model_data = cam_model_data
            .into_iter()
            .map(|cam| (cam.id, cam))
            .collect::<HashMap<_, _>>();
        let mut img_file = vfs.reader_at_path(&img_path).await?;
        let img_infos =
            colmap_reader::read_images(&mut img_file, is_binary, load_args.estimate_metric_scale)
                .await?;
        let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();
        img_info_list.sort_by(|img_a, img_b| img_a.name.cmp(&img_b.name));

        log::info!("Loading {} images for colmap dataset", img_info_list.len());

        let mut warnings = Vec::new();
        // COLMAP is reconstructed up to an unknown global scale. When depth
        // maps are present and the flag is set, recover the scale that lines
        // the landmarks up with those maps. The flag alone does not change
        // the scene.
        let metric_scale = if load_args.estimate_metric_scale {
            match estimate_metric_scale(
                &vfs,
                &img_info_list,
                &cam_model_data,
                &points_dir_for_scale,
            )
            .await
            {
                Ok(scale) => {
                    log::info!(
                        "Rescaling COLMAP reconstruction into depth-map units (scale = {scale})"
                    );
                    Some(scale)
                }
                Err(reason) => {
                    log::warn!("{reason}");
                    warnings.push(reason);
                    None
                }
            }
        } else {
            None
        };
        let pose_scale = metric_scale.unwrap_or(1.0);

        let mut views = Vec::new();

        for img_info in img_info_list
            .iter()
            .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
            .take(load_args.max_frames.unwrap_or(usize::MAX))
        {
            let colmap_camera = cam_model_data
                .get(&img_info.camera_id)
                .ok_or_else(|| {
                    FormatError::InvalidFormat(format!(
                        "Image '{}' references camera ID {} which doesn't exist in camera data",
                        img_info.name, img_info.camera_id
                    ))
                })?
                .clone();

            // Create a future to handle loading the image.
            let camera_model = build_camera_model(&colmap_camera);
            let focal = colmap_camera.focal();
            let fovx = camera::focal_to_fov(focal.0, colmap_camera.width as u32, &camera_model);
            let fovy = camera::focal_to_fov(focal.1, colmap_camera.height as u32, &camera_model);
            let center = colmap_camera.principal_point();
            let center_uv =
                center / glam::vec2(colmap_camera.width as f32, colmap_camera.height as f32);

            let Some(path) = find_image_by_name(&vfs, &img_info.name) else {
                warnings.push(format!("Skipped '{}': image file not found", img_info.name));
                continue;
            };

            let mask_path = find_mask_path(&vfs, path);
            let depth = find_depth_path(&vfs, path)
                .map(|p| LoadDepth::new(vfs.clone(), p.to_path_buf(), load_args.units_per_meter));

            // Convert w2c to c2w. `pose_scale` is 1 unless metric scale
            // estimation produced a finite positive scale.
            let world_to_cam = glam::Affine3A::from_rotation_translation(
                img_info.quat,
                img_info.tvec * pose_scale,
            );
            let cam_to_world = world_to_cam.inverse();
            let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

            let camera = Camera::new(translation, quat, fovx, fovy, center_uv, camera_model);

            if !camera.is_valid() {
                warnings.push(format!(
                    "Skipped '{}': camera contains nan or inf values",
                    img_info.name
                ));
                continue;
            }

            let image = LoadImage::new(
                vfs.clone(),
                path.to_path_buf(),
                mask_path.map(|p| p.to_path_buf()),
                load_args.max_resolution,
                load_args.alpha_mode,
                load_args.invert_masks,
            );
            // Camera metadata is the reconstruction resolution. A resized image
            // file keeps those numbers, and the Mip-Splatting scale floor uses
            // this size as the native pixel focal length. Read the file header
            // (not a full decode) so the cached size matches the pixels.
            let (w, h) = image.dimensions().await?;
            let image_size = glam::uvec2(w, h);

            views.push(SceneView {
                camera,
                image,
                image_size,
                depth,
            });
        }

        let (train_views, eval_views) = split_eval_every(views, load_args.eval_split_every);

        Result::<_, FormatError>::Ok((
            Dataset::from_views(train_views, eval_views),
            warnings,
            metric_scale,
        ))
    });

    let load_args = load_args.clone();

    let init = actor.run(move || async move {
        let (points_path, is_binary) = find_points3d_path(&vfs_init, &points_dir)?;
        // At this point the VFS has said this file exists so just unwrap.
        let mut points_file = vfs_init
            .reader_at_path(points_path)
            .await
            .expect("unreachable");

        let step = load_args.subsample_points.unwrap_or(1) as usize;
        let points_data = colmap_reader::read_points3d(&mut points_file, is_binary, false)
            .await
            .ok()?;

        if points_data.is_empty() {
            return None;
        }

        let positions: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| p.xyz.to_array())
            .collect();
        let colors: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| {
                let sh = rgb_to_sh(glam::vec3(
                    p.rgb[0] as f32 / 255.0,
                    p.rgb[1] as f32 / 255.0,
                    p.rgb[2] as f32 / 255.0,
                ));
                [sh.x, sh.y, sh.z]
            })
            .collect();

        let n_splats = positions.len() / 3;
        log::info!("Starting from colmap points: {n_splats}");
        let data = SplatData {
            means: positions,
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(colors),
            raw_opacities: None,
        };

        Some(SplatMessage {
            meta: ParseMetadata {
                up_axis: None,
                render_mode: None,
                total_splats: n_splats as u32,
                progress: 1.0,
            },
            data,
        })
    });

    // Wait for both halves.
    let (dataset, init) = tokio::join!(dataset, init);
    let ((dataset, warnings, metric_scale), init_splat) = (dataset?, init);

    // The initializer is scaled once the caller has chosen it. A later PLY
    // override is in this same frame and must receive the same scale.
    Ok(DatasetLoadResult {
        init_splat,
        dataset,
        warnings,
        metric_scale,
    })
}

/// Estimate the scale that maps this COLMAP reconstruction into the depth
/// maps' units. Errors are messages: a missing depth folder or a lack of
/// usable landmarks leaves the scene unchanged.
async fn estimate_metric_scale(
    vfs: &Arc<BrushVfs>,
    images: &[colmap_reader::Image],
    cameras: &HashMap<i32, ColmapCamera>,
    points_dir: &Path,
) -> Result<f32, String> {
    let depth_views: Vec<_> = images
        .iter()
        .filter_map(|img| {
            let image_path = find_image_by_name(vfs, &img.name)?;
            let depth_path = find_depth_path(vfs, image_path)?;
            Some((img, depth_path))
        })
        .collect();
    if depth_views.is_empty() {
        return Err(
            "Metric scale estimation was requested, but no depth maps were found. The reconstruction scale was left unchanged."
                .to_owned(),
        );
    }

    let Some((points_path, is_binary)) = find_points3d_path(vfs, points_dir) else {
        return Err(
            "Metric scale estimation found depth maps but no points3D file. The reconstruction scale was left unchanged."
                .to_owned(),
        );
    };
    let mut points_file = vfs
        .reader_at_path(points_path)
        .await
        .map_err(|err| format!("Metric scale estimation could not open points3D: {err}"))?;
    let points = colmap_reader::read_points3d(&mut points_file, is_binary, false)
        .await
        .map_err(|err| format!("Metric scale estimation could not read points3D: {err}"))?;
    let point_map: HashMap<i64, glam::Vec3> = points.iter().map(|p| (p.id, p.xyz)).collect();

    let mut colmap_depths = Vec::new();
    let mut metric_depths = Vec::new();
    for (img, depth_path) in depth_views {
        let Some(point_data) = &img.points else {
            continue;
        };
        // File units, not training metres. `units_per_meter` is applied to
        // the geometry and to the training depth targets afterwards.
        let loader = LoadDepth::new(vfs.clone(), depth_path.to_path_buf(), 1.0);
        let Ok((depth, width, height)) = loader.load_native().await else {
            log::warn!(
                "Skipping depth map {} while estimating scale",
                depth_path.display()
            );
            continue;
        };
        let Some(camera) = cameras.get(&img.camera_id) else {
            continue;
        };
        for (xy, &pid) in point_data.xys.iter().zip(point_data.point3d_ids.iter()) {
            if pid < 0 {
                continue;
            }
            let Some(&xyz) = point_map.get(&pid) else {
                continue;
            };
            // Landmarks are in the COLMAP camera's pixel grid. The depth
            // map may be a different resolution.
            let Some((u, v)) = depth_pixel_for_landmark(
                *xy,
                camera.width as u32,
                camera.height as u32,
                width,
                height,
            ) else {
                continue;
            };
            colmap_depths.push(colmap_camera_depth(img.quat, img.tvec, xyz));
            metric_depths.push(depth[v * width + u]);
        }
    }

    metric_scale_from_pairs(&colmap_depths, &metric_depths).ok_or_else(|| {
        "Metric scale estimation found depth maps but no valid landmark correspondences. The reconstruction scale was left unchanged."
            .to_owned()
    })
}

fn build_camera_model(colmap_camera: &ColmapCamera) -> CameraModel {
    let p = &colmap_camera.params;
    // Param layouts follow COLMAP's `src/colmap/sensor/models.h`. Indices
    // are 0-based positions into `p` after the intrinsics (fx, fy, cx, cy
    // or f, cx, cy depending on the model).
    match colmap_camera.model {
        // No distortion.
        ColmapCameraModel::SimplePinhole | ColmapCameraModel::Pinhole => Pinhole,
        // Pure-radial perspective models → RT8 with the higher-order /
        // tangential coefficients zeroed.
        // SIMPLE_RADIAL: f cx cy k1
        ColmapCameraModel::SimpleRadial => RadialTangential8(RadialTangential8Params {
            k1: p[3] as f32,
            ..Default::default()
        }),
        // RADIAL: f cx cy k1 k2
        ColmapCameraModel::Radial => RadialTangential8(RadialTangential8Params {
            k1: p[3] as f32,
            k2: p[4] as f32,
            ..Default::default()
        }),
        // OPENCV: fx fy cx cy k1 k2 p1 p2 (Brown-Conrady, 4 distortion coefficients).
        ColmapCameraModel::OpenCV => RadialTangential8(RadialTangential8Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            p1: p[6] as f32,
            p2: p[7] as f32,
            ..Default::default()
        }),
        // FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6.
        ColmapCameraModel::FullOpenCV => RadialTangential8(RadialTangential8Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[8] as f32,
            k4: p[9] as f32,
            k5: p[10] as f32,
            k6: p[11] as f32,
            p1: p[6] as f32,
            p2: p[7] as f32,
        }),
        // Fisheye variants → KB4 with unused k's zeroed.
        // SIMPLE_RADIAL_FISHEYE: f cx cy k1
        ColmapCameraModel::SimpleRadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            ..Default::default()
        }),
        // RADIAL_FISHEYE: f cx cy k1 k2
        ColmapCameraModel::RadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            k2: p[4] as f32,
            ..Default::default()
        }),
        // OPENCV_FISHEYE: fx fy cx cy k1 k2 k3 k4
        ColmapCameraModel::OpenCvFishEye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[6] as f32,
            k4: p[7] as f32,
        }),
        // THIN_PRISM_FISHEYE: fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
        ColmapCameraModel::ThinPrismFisheye => ThinPrismFisheye(ThinPrismFisheyeParams {
            kb4: KannalaBrandt4Params {
                k1: p[4] as f32,
                k2: p[5] as f32,
                k3: p[8] as f32,
                k4: p[9] as f32,
            },
            p1: p[6] as f32,
            p2: p[7] as f32,
            sx1: p[10] as f32,
            sy1: p[11] as f32,
        }),
        // FOV uses a tan(ω r) / ω model that doesn't fit either of our
        // distortion polynomials. Fall back to pinhole — rare in practice.
        ColmapCameraModel::Fov => {
            log::warn!("COLMAP `FOV` model is not directly supported; falling back to pinhole.");
            Pinhole
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::config::LoadDatasetConfig;
    use crate::formats::{DatasetLoadResult, load_dataset};
    use brush_render::camera::focal_to_fov;
    use brush_render::kernels::camera_model::CameraModel;
    use brush_render::sh::rgb_to_sh;
    use brush_vfs::BrushVfs;
    use std::path::Path;
    use std::sync::Arc;

    const IMG_W: u32 = 64;
    const IMG_H: u32 = 48;
    const FX: f64 = 80.0;
    const FY: f64 = 70.0;
    const CX: f64 = 30.0;
    const CY: f64 = 20.0;

    /// World-to-cam pose written to images.txt for `img1.png`: a 90° rotation
    /// about Y plus a translation.
    fn img1_w2c() -> (glam::Quat, glam::Vec3) {
        (
            glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            glam::vec3(1.0, 0.0, 2.0),
        )
    }

    /// Write a minimal COLMAP text-format model plus tiny PNG files to `dir`.
    /// `missing.png` is referenced in images.txt but has no image file, so the
    /// loader should skip it with a warning.
    async fn write_test_dataset(dir: &Path) {
        let sparse = dir.join("sparse/0");
        tokio::fs::create_dir_all(&sparse).await.unwrap();

        let cameras = format!(
            "# Camera list with one line of data per camera:\n\
             #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
             1 PINHOLE {IMG_W} {IMG_H} {FX} {FY} {CX} {CY}\n"
        );
        tokio::fs::write(sparse.join("cameras.txt"), cameras)
            .await
            .unwrap();

        let (q, t) = img1_w2c();
        let images = format!(
            "# Image list with two lines of data per image:\n\
             #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
             #   POINTS2D[] as (X, Y, POINT3D_ID)\n\
             1 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 img0.png\n\
             \n\
             2 {} {} {} {} {} {} {} 1 img1.png\n\
             10.0 20.0 1 30.0 40.0 -1\n\
             3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 img2.png\n\
             \n\
             4 1.0 0.0 0.0 0.0 5.0 5.0 5.0 1 missing.png\n\
             \n",
            q.w, q.x, q.y, q.z, t.x, t.y, t.z
        );
        tokio::fs::write(sparse.join("images.txt"), images)
            .await
            .unwrap();

        let points = "# 3D point list with one line of data per point:\n\
                      #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n\
                      1 1.5 2.5 3.5 255 0 0 0.5 1 0\n\
                      2 -1.0 0.0 1.0 0 255 0 0.5 2 1\n\
                      3 0.0 1.0 0.0 0 0 255 0.5 3 0\n\
                      4 2.0 2.0 2.0 128 128 128 0.5 1 1\n";
        tokio::fs::write(sparse.join("points3D.txt"), points)
            .await
            .unwrap();

        let images_dir = dir.join("images");
        tokio::fs::create_dir_all(&images_dir).await.unwrap();
        for name in ["img0.png", "img1.png", "img2.png"] {
            let img = image::RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]));
            img.save(images_dir.join(name)).unwrap();
        }
    }

    fn test_config(eval_split_every: Option<usize>) -> LoadDatasetConfig {
        LoadDatasetConfig {
            max_frames: None,
            max_resolution: 1920,
            eval_split_every,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: None,
            invert_masks: false,
            units_per_meter: 1.0,
            estimate_metric_scale: false,
            max_scene_batch_cache_size: 0,
        }
    }

    async fn load_test_dataset(dir: &Path, eval_split_every: Option<usize>) -> DatasetLoadResult {
        let vfs = Arc::new(BrushVfs::from_path(dir).await.unwrap());
        load_dataset(vfs, &test_config(eval_split_every))
            .await
            .unwrap()
    }

    fn assert_vec3_close(a: glam::Vec3, b: glam::Vec3) {
        assert!((a - b).length() < 1e-4, "expected {b:?}, got {a:?}");
    }

    #[tokio::test]
    async fn loads_text_model() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let result = load_test_dataset(dir.path(), None).await;

        // All three found images are train views; the missing one is skipped
        // with a warning instead of failing the load.
        let views = &result.dataset.train.views;
        assert_eq!(views.len(), 3);
        assert!(result.dataset.eval.is_none());
        // File headers are 4x3; cameras.txt still says 64x48. The cached size
        // follows the file, which is what the scale floor's pixel focal uses.
        assert!(
            views
                .iter()
                .all(|view| view.image_size == glam::uvec2(4, 3))
        );
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("missing.png"));

        // Intrinsics: fov derives from the written focal, and roundtrips back.
        let cam = &views[0].camera;
        assert!(matches!(cam.camera_model, CameraModel::Pinhole));
        let expected_fov_x = focal_to_fov(FX, IMG_W, &CameraModel::Pinhole);
        let expected_fov_y = focal_to_fov(FY, IMG_H, &CameraModel::Pinhole);
        assert!((cam.fov_x - expected_fov_x).abs() < 1e-6);
        assert!((cam.fov_y - expected_fov_y).abs() < 1e-6);
        let focal = cam.focal(glam::uvec2(IMG_W, IMG_H));
        assert!((focal.x - FX as f32).abs() < 1e-3);
        assert!((focal.y - FY as f32).abs() < 1e-3);
        let center = cam.center(glam::uvec2(IMG_W, IMG_H));
        assert!((center.x - CX as f32).abs() < 1e-3);
        assert!((center.y - CY as f32).abs() < 1e-3);

        // Views are sorted by image name. img0 has identity rotation and
        // translation (1, 2, 3) in world-to-cam, so cam-to-world is just the
        // negated translation.
        assert_vec3_close(views[0].camera.position, glam::vec3(-1.0, -2.0, -3.0));
        // Note: angle_between amplifies ulp-level differences (acos near 1),
        // so 1e-3 rad (~0.06 degrees) is as tight as is meaningful here.
        assert!(views[0].camera.rotation.angle_between(glam::Quat::IDENTITY) < 1e-3);
        // img2 sits at the origin.
        assert_vec3_close(views[2].camera.position, glam::Vec3::ZERO);

        // img1 has a non-trivial pose: compare against inverting the same
        // world-to-cam transform that was written to images.txt.
        let (q, t) = img1_w2c();
        let cam_to_world = glam::Affine3A::from_rotation_translation(q, t).inverse();
        let (_, expected_rot, expected_pos) = cam_to_world.to_scale_rotation_translation();
        assert_vec3_close(views[1].camera.position, expected_pos);
        assert!(
            views[1].camera.rotation.angle_between(expected_rot) < 1e-3,
            "expected {expected_rot:?}, got {:?}",
            views[1].camera.rotation
        );
        // Sanity-check the hand-derived value too: -R^T * (1, 0, 2).
        assert_vec3_close(views[1].camera.position, glam::vec3(2.0, 0.0, -1.0));

        // Initial point cloud comes from points3D.txt.
        let init = result.init_splat.expect("expected an initial point cloud");
        assert_eq!(init.meta.total_splats, 4);
        assert_eq!(init.data.means.len(), 4 * 3);
        assert_eq!(&init.data.means[0..3], &[1.5, 2.5, 3.5]);
        let sh_coeffs = init.data.sh_coeffs.expect("expected sh coefficients");
        assert_eq!(sh_coeffs.len(), 4 * 3);
        let expected_sh = rgb_to_sh(glam::vec3(1.0, 0.0, 0.0));
        assert_eq!(&sh_coeffs[0..3], expected_sh.to_array().as_slice());
    }

    #[tokio::test]
    async fn units_per_meter_scales_cameras_and_init_points() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.unwrap());
        let base = load_test_dataset(dir.path(), Some(2)).await;
        let config = LoadDatasetConfig {
            units_per_meter: 100.0,
            ..test_config(Some(2))
        };
        let scaled = load_dataset(vfs, &config).await.unwrap();

        // Train and eval camera positions shrink; orientation and intrinsics don't.
        let scenes = |r: &DatasetLoadResult| {
            let mut v = r.dataset.train.views.as_ref().clone();
            v.extend(r.dataset.eval.as_ref().unwrap().views.iter().cloned());
            v
        };
        for (a, b) in scenes(&base).into_iter().zip(scenes(&scaled)) {
            assert_vec3_close(b.camera.position, a.camera.position / 100.0);
            assert_eq!(b.camera.rotation, a.camera.rotation);
            assert_eq!(b.camera.fov_x, a.camera.fov_x);
        }

        // Initial point cloud shrinks along with it.
        let init = scaled.init_splat.expect("expected an initial point cloud");
        assert_vec3_close(
            glam::Vec3::from_slice(&init.data.means[0..3]),
            glam::vec3(0.015, 0.025, 0.035),
        );
        if let Some(log_scales) = init.data.log_scales {
            let base_scales = base.init_splat.unwrap().data.log_scales.unwrap();
            for (a, b) in base_scales.iter().zip(&log_scales) {
                assert!((b - (a - 100f32.ln())).abs() < 1e-5);
            }
        }

        // A non-positive factor is rejected instead of producing NaNs.
        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.unwrap());
        let bad = LoadDatasetConfig {
            units_per_meter: 0.0,
            ..test_config(None)
        };
        assert!(load_dataset(vfs, &bad).await.is_err());
    }

    #[tokio::test]
    async fn splits_eval_views() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let result = load_test_dataset(dir.path(), Some(2)).await;

        // Every 2nd view (indices 0 and 2) goes to eval, the rest to train.
        let train = &result.dataset.train.views;
        let eval = &result
            .dataset
            .eval
            .as_ref()
            .expect("expected eval split")
            .views;
        assert_eq!(train.len(), 1);
        assert_eq!(eval.len(), 2);

        // Identify views by their distinct camera positions: img1 (the
        // rotated view) trains, img0 and img2 evaluate.
        let (q, t) = img1_w2c();
        let expected_train_pos = glam::Affine3A::from_rotation_translation(q, t)
            .inverse()
            .translation
            .into();
        assert_vec3_close(train[0].camera.position, expected_train_pos);
        assert_vec3_close(eval[0].camera.position, glam::vec3(-1.0, -2.0, -3.0));
        assert_vec3_close(eval[1].camera.position, glam::Vec3::ZERO);
    }

    /// Identity camera at z=5 looking at a point at z=10 (COLMAP depth 5).
    /// The depth map says 10 at the landmark pixel, so the scale is 2.
    async fn write_metric_depth_dataset(dir: &Path) {
        let sparse = dir.join("sparse/0");
        tokio::fs::create_dir_all(&sparse).await.unwrap();
        tokio::fs::write(
            sparse.join("cameras.txt"),
            "# Camera list with one line of data per camera:\n\
             #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
             1 PINHOLE 4 3 4.0 4.0 2.0 1.5\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            sparse.join("images.txt"),
            "# Image list with two lines of data per image:\n\
             #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
             #   POINTS2D[] as (X, Y, POINT3D_ID)\n\
             1 1 0 0 0 0 0 -5 1 view.png\n\
             1 1 1\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            sparse.join("points3D.txt"),
            "# 3D point list with one line of data per point:\n\
             #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n\
             1 0 0 10 255 0 0 0.1 1 0\n",
        )
        .await
        .unwrap();

        let images = dir.join("images");
        tokio::fs::create_dir_all(&images).await.unwrap();
        image::RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]))
            .save(images.join("view.png"))
            .unwrap();

        let mut samples = vec![0.0f32; 4 * 3];
        // Pixel (1, 1) of the 4-wide depth map, the landmark's nearest sample.
        samples[5] = 10.0;
        let mut bytes = std::io::Cursor::new(Vec::new());
        tiff::encoder::TiffEncoder::new(&mut bytes)
            .unwrap()
            .write_image::<tiff::encoder::colortype::Gray32Float>(4, 3, &samples)
            .unwrap();
        let bytes = bytes.into_inner();
        let depth_dir = dir.join("depth");
        tokio::fs::create_dir_all(&depth_dir).await.unwrap();
        tokio::fs::write(depth_dir.join("view.tiff"), bytes)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn loads_depth_maps_and_can_estimate_metric_scale() {
        let dir = tempfile::tempdir().unwrap();
        write_metric_depth_dataset(dir.path()).await;

        let unscaled = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &test_config(None),
        )
        .await
        .unwrap();
        let view = &unscaled.dataset.train.views[0];
        assert!(view.depth.is_some(), "depth map should be attached");
        assert_vec3_close(view.camera.position, glam::vec3(0.0, 0.0, 5.0));
        let depth = view
            .depth
            .as_ref()
            .unwrap()
            .load_resized(3, 4)
            .await
            .unwrap();
        let samples = depth.try_to_vec::<f32>().unwrap();
        assert!((samples[5] - 10.0).abs() < 1e-5);

        let scaled = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &LoadDatasetConfig {
                estimate_metric_scale: true,
                ..test_config(None)
            },
        )
        .await
        .unwrap();
        assert!(
            scaled.warnings.is_empty(),
            "unexpected warnings: {:?}",
            scaled.warnings
        );
        assert_vec3_close(
            scaled.dataset.train.views[0].camera.position,
            glam::vec3(0.0, 0.0, 10.0),
        );
        let init = scaled.init_splat.expect("initial points");
        assert_vec3_close(
            glam::Vec3::from_slice(&init.data.means[0..3]),
            glam::vec3(0.0, 0.0, 20.0),
        );
    }

    #[tokio::test]
    async fn estimate_metric_scale_without_depth_leaves_the_scene_alone() {
        let dir = tempfile::tempdir().unwrap();
        write_test_dataset(dir.path()).await;
        let result = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &LoadDatasetConfig {
                estimate_metric_scale: true,
                ..test_config(None)
            },
        )
        .await
        .unwrap();
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("no depth maps")),
            "warnings: {:?}",
            result.warnings
        );
        assert!(
            result
                .dataset
                .train
                .views
                .iter()
                .all(|view| view.depth.is_none())
        );
        assert_vec3_close(
            result.dataset.train.views[0].camera.position,
            glam::vec3(-1.0, -2.0, -3.0),
        );
    }

    /// Camera is 8×6. The landmark sits at camera pixel (2, 2), which is
    /// depth pixel (1, 1) on the 4×3 map. Indexing the map with (2, 2) would
    /// read a zero and drop the correspondence.
    #[tokio::test]
    async fn metric_scale_samples_landmarks_in_depth_map_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let sparse = dir.path().join("sparse/0");
        tokio::fs::create_dir_all(&sparse).await.unwrap();
        tokio::fs::write(
            sparse.join("cameras.txt"),
            "# Camera list with one line of data per camera:\n\
             #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
             1 PINHOLE 8 6 8.0 8.0 4.0 3.0\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            sparse.join("images.txt"),
            "# Image list with two lines of data per image:\n\
             #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\
             #   POINTS2D[] as (X, Y, POINT3D_ID)\n\
             1 1 0 0 0 0 0 -5 1 view.png\n\
             2 2 1\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            sparse.join("points3D.txt"),
            "# 3D point list with one line of data per point:\n\
             #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n\
             1 0 0 10 255 0 0 0.1 1 0\n",
        )
        .await
        .unwrap();
        let images = dir.path().join("images");
        tokio::fs::create_dir_all(&images).await.unwrap();
        image::RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]))
            .save(images.join("view.png"))
            .unwrap();
        let mut samples = vec![0.0f32; 4 * 3];
        samples[5] = 10.0;
        let mut bytes = std::io::Cursor::new(Vec::new());
        tiff::encoder::TiffEncoder::new(&mut bytes)
            .unwrap()
            .write_image::<tiff::encoder::colortype::Gray32Float>(4, 3, &samples)
            .unwrap();
        let depth_dir = dir.path().join("depth");
        tokio::fs::create_dir_all(&depth_dir).await.unwrap();
        tokio::fs::write(depth_dir.join("view.tiff"), bytes.into_inner())
            .await
            .unwrap();

        let scaled = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &LoadDatasetConfig {
                estimate_metric_scale: true,
                ..test_config(None)
            },
        )
        .await
        .unwrap();
        assert!(
            scaled.warnings.is_empty(),
            "warnings: {:?}",
            scaled.warnings
        );
        assert_vec3_close(
            scaled.dataset.train.views[0].camera.position,
            glam::vec3(0.0, 0.0, 10.0),
        );
    }

    #[tokio::test]
    async fn depth_targets_follow_units_per_meter() {
        let dir = tempfile::tempdir().unwrap();
        write_metric_depth_dataset(dir.path()).await;
        // Overwrite the depth map with a millimetre reading of the same pixel.
        let mut samples = vec![0.0f32; 4 * 3];
        samples[5] = 1000.0;
        let mut bytes = std::io::Cursor::new(Vec::new());
        tiff::encoder::TiffEncoder::new(&mut bytes)
            .unwrap()
            .write_image::<tiff::encoder::colortype::Gray32Float>(4, 3, &samples)
            .unwrap();
        tokio::fs::write(dir.path().join("depth/view.tiff"), bytes.into_inner())
            .await
            .unwrap();

        let result = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &LoadDatasetConfig {
                estimate_metric_scale: true,
                units_per_meter: 1000.0,
                ..test_config(None)
            },
        )
        .await
        .unwrap();
        assert!(
            result.warnings.is_empty(),
            "warnings: {:?}",
            result.warnings
        );
        // Raw scale is 1000/5 = 200, then positions are divided by 1000.
        assert_vec3_close(
            result.dataset.train.views[0].camera.position,
            glam::vec3(0.0, 0.0, 1.0),
        );
        let init = result.init_splat.expect("initial points");
        assert_vec3_close(
            glam::Vec3::from_slice(&init.data.means[0..3]),
            glam::vec3(0.0, 0.0, 2.0),
        );
        let depth = result.dataset.train.views[0]
            .depth
            .as_ref()
            .unwrap()
            .load_resized(3, 4)
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            (depth[5] - 1.0).abs() < 1e-5,
            "training depth should be in metres, got {}",
            depth[5]
        );
    }

    /// `init.ply` is in the original reconstruction frame, so the public loader
    /// must scale it after preferring it over `points3D`.
    #[tokio::test]
    async fn metric_scale_reaches_an_overriding_ply() {
        let dir = tempfile::tempdir().unwrap();
        write_metric_depth_dataset(dir.path()).await;
        let ply = "\
ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
end_header
0 0 10 -1 -1 -1
";
        tokio::fs::write(dir.path().join("init.ply"), ply)
            .await
            .unwrap();

        let result = load_dataset(
            Arc::new(BrushVfs::from_path(dir.path()).await.unwrap()),
            &LoadDatasetConfig {
                estimate_metric_scale: true,
                ..test_config(None)
            },
        )
        .await
        .unwrap();
        assert_vec3_close(
            result.dataset.train.views[0].camera.position,
            glam::vec3(0.0, 0.0, 10.0),
        );
        let init = result.init_splat.expect("ply initializer");
        assert_vec3_close(
            glam::Vec3::from_slice(&init.data.means[0..3]),
            glam::vec3(0.0, 0.0, 20.0),
        );
        let log_scales = init.data.log_scales.expect("ply log-scales");
        let shift = 2.0f32.ln();
        for scale in log_scales {
            assert!(
                (scale - (-1.0 + shift)).abs() < 1e-5,
                "log-scale {scale}, expected {}",
                -1.0 + shift
            );
        }
    }
}
