use std::path::{Path, PathBuf};

use anyhow::Context;
use brush_dataset::config::DepthMode;
use brush_dataset::scene::{SceneBatch, SceneView};
use brush_dataset::scene_loader::SceneLoader;
use brush_process::config::TrainStreamConfig;
use brush_process::{DataSource, ProcessDevice, burn_init_setup};
use brush_render::AlphaMode;
use brush_render::gaussian_splats::{SplatRenderMode, Splats, render_splats_depth};
use brush_train::train::{BOUND_PERCENTILE, REFINE_STOP_PROGRESS, SplatTrainer, get_splat_bounds};
use burn::module::Module;
use clap::{Parser, ValueEnum};
use glam::Vec3;
use serde_json::json;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[value(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
enum Case {
    Baseline,
    Metric,
    Relative,
}

const DEFAULT_CASES: &[Case] = &[Case::Baseline, Case::Baseline, Case::Metric, Case::Relative];

#[derive(Parser)]
#[command(about = "Compare depth supervision modes on the Issue #7 synthetic dataset")]
struct Args {
    /// Fixture output from `test_cases/issue7/generate.py`.
    fixture: PathBuf,
    /// Selected cases, in order. Omit to run all comparisons and repeat RGB.
    cases: Vec<Case>,
}

async fn metrics(
    splats: &Splats,
    eval: &[SceneView],
    metric_root: &Path,
) -> anyhow::Result<serde_json::Value> {
    let mut foreground = 0;
    let mut covered = 0;
    let mut background = 0;
    let mut outside_alpha = 0.0_f64;
    let mut squared_rgb = 0.0_f64;
    let mut depth_error = 0.0_f64;
    let mut edge_error = 0.0_f64;
    let mut edge_pixels = 0;
    let mut visible = 0;
    let mut held_out = vec![];
    let metric_config = TrainStreamConfig::default().load_config;
    let metric_vfs = DataSource::Path(metric_root.display().to_string())
        .into_vfs()
        .await?;
    let metric_data = brush_dataset::load_dataset(metric_vfs, &metric_config).await?;
    for view in eval {
        held_out.push(view.image.path().file_name().unwrap().to_string_lossy());
        let metric = metric_data
            .dataset
            .train
            .views
            .iter()
            .find(|candidate| candidate.image.path() == view.image.path())
            .with_context(|| {
                format!(
                    "missing metric reference for {}",
                    view.image.path().display()
                )
            })?;
        let (depths, _, _) = metric
            .depth
            .as_ref()
            .context("missing metric reference TIFF")?
            .load_values()
            .await?;
        let img = view.image.load().await?.into_rgba8();
        let (prediction, aux) = render_splats_depth(
            splats.clone(),
            &view.camera,
            view.image_size,
            Vec3::ZERO,
            None,
        )
        .await;
        visible += aux.num_visible;
        let prediction = prediction.into_data_async().await?.try_into_vec::<f32>()?;
        assert_eq!(
            prediction.len(),
            depths.len() * 5,
            "expected RGB, alpha and depth samples"
        );
        assert_eq!(
            depths.len(),
            img.pixels().len(),
            "RGB and depth sizes must match"
        );
        let (prediction, _) = prediction.as_chunks::<5>();
        for (index, (real, pred)) in img.pixels().zip(prediction).enumerate() {
            assert!(
                pred.iter().all(|x| x.is_finite()),
                "non-finite prediction at pixel {index}"
            );
            if real[3] == 0 {
                background += 1;
                outside_alpha += f64::from(pred[3]);
                continue;
            }
            foreground += 1;
            for channel in 0..3 {
                squared_rgb += f64::from(pred[channel] - f32::from(real[channel]) / 255.0).powi(2);
            }
            if pred[3] >= 0.25 {
                depth_error += f64::from((pred[4] / pred[3] - depths[index]).abs());
                covered += 1;
            }
            let x = index as u32 % img.width();
            let y = index as u32 / img.width();
            if x + 1 < img.width() && img.get_pixel(x + 1, y).0[3] > 0 {
                let right = img.get_pixel(x + 1, y);
                for channel in 0..3 {
                    let real_gradient =
                        (f32::from(right.0[channel]) - f32::from(real[channel])) / 255.0;
                    let pred_gradient = prediction[index + 1][channel] - pred[channel];
                    edge_error += f64::from((real_gradient - pred_gradient).abs());
                }
                edge_pixels += 1;
            }
        }
    }
    assert_eq!(
        held_out,
        ["view_00.png", "view_04.png"],
        "held-out views changed"
    );
    assert!(
        covered > 0 && background > 0 && edge_pixels > 0 && visible > 0,
        "incomplete evaluation coverage"
    );
    Ok(json!({
        "held_out": held_out, "visible": visible,
        "outside_alpha": outside_alpha / f64::from(background),
        "coverage": f64::from(covered) / f64::from(foreground),
        "foreground_psnr": -10.0 * (squared_rgb / f64::from(foreground * 3)).log10(),
        "covered_depth_mae": depth_error / f64::from(covered),
        "foreground_horizontal_gradient_mae": edge_error / f64::from(edge_pixels * 3),
    }))
}

async fn run(root: &Path, case: Case, device: &ProcessDevice) -> anyhow::Result<()> {
    let variant = match case {
        Case::Relative => "relative",
        Case::Baseline | Case::Metric => "metric",
    };
    let mut cfg = TrainStreamConfig::default();
    cfg.load_config.eval_split_every = Some(4);
    cfg.load_config.alpha_mode = Some(AlphaMode::Masked);
    if case == Case::Relative {
        cfg.load_config.depth_mode = DepthMode::RelativeInverse;
    }
    let config = &mut cfg.train_config;
    config.total_train_iters = 180;
    config.lr_mean = 0.002;
    config.lr_mean_end = 0.0004;
    config.background_noise_strength = 0.0;
    config.mean_noise_weight = 0.0;
    config.refine_every = 12;
    config.max_splats = 2400;
    config.split_at_screen_size = 0.09;
    config.depth_loss_weight = match case {
        Case::Relative => 0.15,
        Case::Metric => 0.50,
        Case::Baseline => 0.0,
    };

    let vfs = DataSource::Path(root.join(variant).display().to_string())
        .into_vfs()
        .await?;
    let result = brush_dataset::load_dataset(vfs, &cfg.load_config).await?;
    let dataset = result.dataset;
    let mut splats = brush_train::to_init_splats(
        result.init_splat.context("missing fixture init.ply")?.data,
        SplatRenderMode::Default,
        device,
    );
    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new_seeded(config, device, bounds, 42);
    assert_eq!(dataset.train.views.len(), 6, "expected six training views");
    let eval = dataset
        .eval
        .as_ref()
        .context("missing held-out fixture views")?;
    let cams: Vec<_> = dataset
        .train
        .views
        .iter()
        .map(|view| (view.camera.position, view.camera.focal(view.image_size).x))
        .collect();
    trainer.set_view_cams(cams);

    let mut batches: Vec<Option<SceneBatch>> = vec![None; dataset.train.views.len()];
    let mut loader = SceneLoader::new(&dataset.train, 42, &cfg.load_config);
    while batches.iter().any(Option::is_none) {
        let batch = loader.next_batch().await;
        let index = dataset
            .train
            .views
            .iter()
            .position(|view| view.camera == batch.camera)
            .context("loaded camera is absent from the training split")?;
        batches[index] = Some(batch);
    }
    let batches: Vec<_> = batches.into_iter().map(Option::unwrap).collect();
    assert!(
        batches.iter().all(|batch| batch.depth.is_some()),
        "missing training TIFF"
    );
    drop(loader);
    let mut split = 0;
    let mut pruned = 0;
    for iter in 0..config.total_train_iters {
        let batch = batches[iter as usize % batches.len()].clone();
        let (updated, stats) = trainer.step(batch, splats.train()).await;
        splats = updated.valid();
        assert!(
            stats.num_visible > 0,
            "no visible splats at iteration {iter}"
        );
        let phase_progress = iter as f32 / config.total_train_iters as f32;
        if iter > 0
            && iter.is_multiple_of(config.refine_every)
            && phase_progress <= REFINE_STOP_PROGRESS
        {
            let (updated, refine) = trainer.refine(iter, splats).await;
            splats = updated;
            split += refine.num_added;
            pruned += refine.num_pruned;
            assert_eq!(
                refine.num_pruned_non_finite, 0,
                "non-finite splats at iteration {iter}"
            );
        }
    }
    assert!(split > 0, "training must exercise splitting");
    println!(
        "{}",
        json!({
            "case": case, "backend": format!("{:?}", device.as_dispatch()),
            "steps": config.total_train_iters, "splats": splats.num_splats(), "split": split,
            "pruned": pruned,
            "metrics": metrics(&splats, &eval.views, &root.join("metric")).await?,
        })
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cases = if args.cases.is_empty() {
        DEFAULT_CASES
    } else {
        &args.cases
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let device = burn_init_setup().await;
            for &case in cases {
                run(&args.fixture, case, &device).await?;
            }
            Ok(())
        })
}
