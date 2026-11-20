//! Float32 TIFF depth maps. A value of `0` marks an invalid sample.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use thiserror::Error;
use tokio::io::AsyncReadExt;

use crate::depth::resample_nearest;

/// Lazily loaded per-view depth map.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

impl LoadDepth {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf) -> Self {
        Self { vfs, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Decode the file at its native resolution: `(samples, width, height)`.
    pub async fn load_native(&self) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
        let mut bytes = Vec::new();
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;
        decode_f32_tiff(&bytes)
    }

    /// Decode and nearest-resample to `[height, width]`, matching a training image.
    pub async fn load_resized(
        &self,
        height: usize,
        width: usize,
    ) -> Result<TensorData, LoadDepthError> {
        if height == 0 || width == 0 {
            return Err(LoadDepthError::ReadTiffError(
                "depth target size must be non-zero".to_owned(),
            ));
        }
        let (samples, src_w, src_h) = self.load_native().await?;
        let samples = resample_nearest(&samples, src_w, src_h, width, height);
        Ok(TensorData::new(samples, [height, width]))
    }
}

/// Decode a single-channel float32 TIFF in row-major order.
pub fn decode_f32_tiff(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;
    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let tiff::decoder::DecodingResult::F32(depth) = decoder.read_image()? else {
        return Err(LoadDepthError::ReadTiffError(
            "unsupported TIFF sample format (expected float32 depth)".to_owned(),
        ));
    };
    if w == 0 || h == 0 || w * h != depth.len() {
        return Err(LoadDepthError::ReadTiffError(
            "expected a single-channel float32 image".to_owned(),
        ));
    }
    Ok((depth, w, h))
}

#[derive(Error, Debug)]
pub enum LoadDepthError {
    #[error("I/O error while loading depth map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while loading TIFF file: {0}")]
    LoadTiffError(#[from] tiff::TiffError),

    #[error("Error while reading TIFF file: {0}")]
    ReadTiffError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn encode_gray_f32(width: u32, height: u32, data: &[f32]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        tiff::encoder::TiffEncoder::new(&mut cursor)
            .unwrap()
            .write_image::<tiff::encoder::colortype::Gray32Float>(width, height, data)
            .unwrap();
        cursor.into_inner()
    }

    #[test]
    fn decodes_float32_tiff() {
        let samples = [0.0, 1.5, 2.0, 4.0];
        let bytes = encode_gray_f32(2, 2, &samples);
        let (decoded, w, h) = decode_f32_tiff(&bytes).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(decoded, samples);
    }

    #[test]
    fn rejects_non_float_tiff() {
        let mut cursor = Cursor::new(Vec::new());
        tiff::encoder::TiffEncoder::new(&mut cursor)
            .unwrap()
            .write_image::<tiff::encoder::colortype::Gray8>(2, 1, &[1, 2])
            .unwrap();
        let err = decode_f32_tiff(&cursor.into_inner()).unwrap_err();
        assert!(
            err.to_string().contains("float32"),
            "unexpected error: {err}"
        );
    }
}
