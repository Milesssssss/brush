#![cfg(not(target_family = "wasm"))]

use std::path::Path;

use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::{DataSource, RunningProcess, burn_init_setup, create_process_with_device};
use tokio_stream::StreamExt;

fn config(output: &Path) -> TrainStreamConfig {
    let mut config = TrainStreamConfig::default();
    config.train_config.total_train_iters = 21;
    config.train_config.refine_every = 5;
    config.train_config.growth_stop_iter = 0;
    config.train_config.mean_noise_weight = 0.0;
    config.model_config.sh_degree = 0;
    config.load_config.max_resolution = 50;
    config.process_config.export_path = output.to_str().unwrap().to_owned();
    config
}

async fn start(config: TrainStreamConfig) -> RunningProcess {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/test_dataset");
    let device = burn_init_setup().await;
    create_process_with_device(
        DataSource::Path(path.to_str().unwrap().to_owned()),
        device,
        |_| async { Some(config) },
    )
}

async fn effective_opacities(run: &RunningProcess) -> Vec<f32> {
    let splats = run.splat_view.latest().unwrap();
    let data = splats
        .bake_min_scale()
        .opacities()
        .into_data_async()
        .await
        .unwrap();
    data.as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn reset_runs_after_the_first_refine_when_resuming() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let mut config = config(dir.path());
            config.train_config.split_at_screen_size = 0.01;
            config.train_config.opacity_reset_every = 5;
            config.process_config.checkpoint_every = 5;
            let mut run = start(config.clone()).await;
            while let Some(event) = run.stream.next().await {
                if matches!(
                    event.unwrap(),
                    ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter: 5, .. })
                ) {
                    break;
                }
            }
            let before = effective_opacities(&run).await;
            assert!(before.iter().any(|alpha| *alpha > 0.1));
            let checkpoint = dir.path().join("checkpoint.safetensors");
            assert!(checkpoint.is_file());
            drop(run);

            config.process_config.resume = Some(checkpoint.to_str().unwrap().to_owned());
            let mut resumed = start(config).await;
            let mut checkpoint_loaded = false;
            let mut reset_observed = false;
            while let Some(event) = resumed.stream.next().await {
                match event.unwrap() {
                    ProcessMessage::DoneLoading => {
                        assert_eq!(effective_opacities(&resumed).await, before);
                        checkpoint_loaded = true;
                    }
                    ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                        iter,
                        cur_splat_count,
                    }) => {
                        assert!(checkpoint_loaded);
                        assert_eq!(iter, 6);
                        assert!(cur_splat_count as usize > before.len());
                        assert!(resumed.splat_view.latest().unwrap().min_scale.is_some());
                        let effective = effective_opacities(&resumed).await;
                        assert_eq!(effective.len(), cur_splat_count as usize);
                        assert!(
                            effective
                                .iter()
                                .all(|alpha| alpha.is_finite() && *alpha <= 0.01001)
                        );
                        assert!(effective.iter().any(|alpha| *alpha >= 0.00999));
                        reset_observed = true;
                    }
                    ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter: 10, .. }) => {
                        assert!(reset_observed);
                        assert!(
                            effective_opacities(&resumed)
                                .await
                                .iter()
                                .all(|alpha| alpha.is_finite())
                        );
                        return;
                    }
                    _ => {}
                }
            }
            panic!("the resumed fixture must continue past the split and reset boundary");
        });
}

#[test]
fn reset_does_not_fire_past_the_phase_cutoff() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let mut config = config(dir.path());
            config.train_config.split_at_screen_size = 0.0;
            config.train_config.lr_opac = 0.0;
            config.train_config.opacity_reset_every = 20;
            let mut run = start(config).await;
            while let Some(event) = run.stream.next().await {
                if matches!(
                    event.unwrap(),
                    ProcessMessage::TrainMessage(TrainMessage::DoneTraining)
                ) {
                    let effective = effective_opacities(&run).await;
                    assert!(effective.iter().any(|alpha| *alpha > 0.05));
                    return;
                }
            }
            panic!("training did not reach the phase cutoff");
        });
}
