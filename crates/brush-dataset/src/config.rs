use brush_render::AlphaMode;
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

/// Default Cache budget for packed scene batches. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "6GiB";
#[cfg(target_family = "wasm")]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "2GiB";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DepthMode {
    /// Camera Z in the dataset's units. Only finite positive values are valid.
    #[default]
    Metric,
    /// Unitless inverse depth with larger values closer. All finite values are valid.
    RelativeInverse,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(
        long,
        help_heading = "Model Options",
        default_value = "3",
        value_parser = clap::value_parser!(u32).range(0..=4)
    )]
    pub sh_degree: u32,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LoadDatasetConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    #[arg(long, help_heading = "Dataset Options", default_value = "1920")]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
    /// Invert mask images, so white means "ignore this pixel" instead of "keep it".
    #[arg(long, help_heading = "Dataset Options", default_value = "false")]
    pub invert_masks: bool,
    /// Dataset units per metre. Camera positions and initial splats are divided
    /// by this on load so training runs in metres, and exports multiply back.
    /// E.g. 1000 for a scene reconstructed in millimetres. Applied after
    /// `--estimate-metric-scale` when that flag recovers a scale.
    #[arg(long, help_heading = "Dataset Options", default_value = "1.0")]
    pub units_per_meter: f32,
    /// Meaning of the float32 TIFF values in the COLMAP depth directory.
    #[arg(long, help_heading = "Dataset Options", default_value = "metric")]
    #[serde(default)]
    pub depth_mode: DepthMode,
    /// Recover one global scale from COLMAP landmarks and per-image depth maps
    /// so the reconstruction matches the depth maps' units. Off by default,
    /// and skipped when no `depth` folder is present.
    #[arg(long, help_heading = "Dataset Options", default_value_t = false)]
    #[serde(default)]
    pub estimate_metric_scale: bool,
    /// Max size of the cache for frames of the dataset, larger values usually improve performance for large datasets at the cost of more memory usage, can be e.g. 6G, 6000M, 6000MiB, 6000MB
    #[arg(long, help_heading = "Dataset Options", default_value = DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE, value_parser = parse_size)]
    pub max_scene_batch_cache_size: u64,
}

fn parse_size(s: &str) -> Result<u64, parse_size::Error> {
    parse_size::parse_size(s)
}
