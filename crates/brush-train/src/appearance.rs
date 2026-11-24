use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::tensor::{Device, Gradients, Int, Tensor, TensorData, s};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::adam_scaled::{AdamScaled, AdamState, MomentumState};

const LEARNING_RATE: f64 = 0.01;
pub(crate) const IDENTITY_WEIGHT: f32 = 0.01;

#[derive(Serialize, Deserialize)]
pub(crate) struct AppearanceViews {
    pub paths: Vec<PathBuf>,
    pub steps: Vec<usize>,
}

pub(crate) struct Appearance {
    pub views: AppearanceViews,
    pub params: Tensor<3>,
    pub moment_1: Tensor<3>,
    pub moment_2: Tensor<3>,
}

impl Appearance {
    pub fn new(paths: Vec<PathBuf>, device: &Device) -> Self {
        let n = paths.len();
        Self {
            views: AppearanceViews {
                paths,
                steps: vec![0; n],
            },
            params: Tensor::zeros([n, 4, 3], device),
            moment_1: Tensor::zeros([n, 4, 3], device),
            moment_2: Tensor::zeros([n, 4, 3], device),
        }
    }

    pub fn set_views(&mut self, paths: Vec<PathBuf>) {
        if self.views.paths == paths {
            return;
        }

        let by_path: HashMap<_, _> = self.views.paths.iter().zip(0..).collect();
        let new_view = self.views.paths.len();
        let old_rows: Vec<usize> = paths
            .iter()
            .map(|path| by_path.get(path).copied().unwrap_or(new_view))
            .collect();
        let device = self.params.device();
        let rows = Tensor::<1, Int>::from_data(
            TensorData::new(old_rows.iter().map(|&i| i as i32).collect(), [paths.len()]),
            &device,
        );
        let zeros = Tensor::zeros([1, 4, 3], &device);
        let remap = |table| Tensor::cat(vec![table, zeros.clone()], 0).select(0, rows.clone());
        self.params = remap(self.params.clone());
        self.moment_1 = remap(self.moment_1.clone());
        self.moment_2 = remap(self.moment_2.clone());
        self.views.steps = old_rows
            .iter()
            .map(|&i| self.views.steps.get(i).copied().unwrap_or(0))
            .collect();
        self.views.paths = paths;
    }

    pub fn for_training(&self, index: usize) -> Tensor<2> {
        Tensor::from_inner(affine_row(&self.params, index)).require_grad()
    }

    pub fn for_view(&self, path: &Path) -> Option<Tensor<2>> {
        let index = self.views.paths.iter().position(|p| p == path)?;
        (self.views.steps[index] > 0).then(|| affine_row(&self.params, index))
    }

    pub fn update(&mut self, index: usize, delta: Tensor<2>, grads: &mut Gradients) {
        let grad = delta
            .grad_remove(grads)
            .expect("appearance correction is part of the image loss")
            .without_autodiff();
        let time = self.views.steps[index];
        let mut state = AdamState::new(None, false);
        if time > 0 {
            state.momentum = Some(MomentumState {
                moment_1: affine_row(&self.moment_1, index),
                moment_2: affine_row(&self.moment_2, index),
                time,
            });
        }
        let updated =
            AdamScaled::new(1e-8).step(LEARNING_RATE, delta.inner(), &grad, None, &mut state);
        let moment = state.momentum.expect("Adam records moments on each step");
        self.views.steps[index] = moment.time;
        let update_row = |table: Tensor<3>, row: Tensor<2>| {
            table.slice_assign(s![index..index + 1, .., ..], row.unsqueeze_dim(0))
        };
        self.params = update_row(self.params.clone(), updated);
        self.moment_1 = update_row(self.moment_1.clone(), moment.moment_1);
        self.moment_2 = update_row(self.moment_2.clone(), moment.moment_2);
    }
}

fn affine_row(table: &Tensor<3>, index: usize) -> Tensor<2> {
    table
        .clone()
        .slice(s![index..index + 1, .., ..])
        .reshape([4, 3])
}

pub(crate) fn correct_image(rgba: Tensor<3>, delta: Tensor<2>, background: Vec3) -> Tensor<3> {
    let [h, w, _] = rgba.dims();
    let bg = Tensor::<1>::from_floats(background.to_array(), &rgba.device()).reshape([1, 1, 3]);
    let rgb = rgba.clone().slice(s![.., .., ..3]);
    let alpha = rgba.slice(s![.., .., 3..4]);
    // Rasterized RGB includes synthetic background and premultiplied foreground.
    // Apply the matrix and bias to foreground coverage before adding the residual.
    let transmittance: Tensor<3> = 1.0 - alpha.clone();
    let foreground = rgb.clone() - transmittance * bg;
    let residual = Tensor::cat(vec![foreground, alpha.clone()], 2)
        .reshape([h * w, 4])
        .matmul(delta)
        .reshape([h, w, 3]);
    Tensor::cat(vec![rgb + residual, alpha], 2)
}

#[cfg(test)]
mod tests;
