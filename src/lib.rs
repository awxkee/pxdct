/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod dct2;
mod dct3;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod pxdct_error;
mod spectrum_mul;
mod twiddles;
mod util;

use crate::dct2::Dct2Fft;
use crate::dct3::Dct3Fft;
pub use pxdct_error::PxdctError;

/// The main entry point for creating DCT (Discrete Cosine Transform) executors.
///
/// `Pxdct` provides convenient factory methods to construct optimized
/// executors for DCT-II and DCT-III transforms using single (`f32`) or
/// double (`f64`) precision. Each executor implements the [`PxdctExecutor`]
/// trait and can be used to perform an in-place DCT transform on a data slice.
pub struct Pxdct {}

impl Pxdct {
    /// Creates a single-precision (f32) DCT-II executor.
    pub fn make_dct2_f32(
        length: usize,
    ) -> Result<Box<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Dct2Fft::new(length).map(|x| Box::new(x) as Box<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DCT-II executor.
    pub fn make_dct2_f64(
        length: usize,
    ) -> Result<Box<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Dct2Fft::new(length).map(|x| Box::new(x) as Box<dyn PxdctExecutor<f64> + Send + Sync>)
    }

    /// Creates a single-precision (f32) DCT-III executor.
    pub fn make_dct3_f32(
        length: usize,
    ) -> Result<Box<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Dct3Fft::new(length).map(|x| Box::new(x) as Box<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DCT-III executor.
    pub fn make_dct3_f64(
        length: usize,
    ) -> Result<Box<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Dct3Fft::new(length).map(|x| Box::new(x) as Box<dyn PxdctExecutor<f64> + Send + Sync>)
    }
}

/// Trait implemented by all PXDCT executors.
///
/// This trait defines the common interface for performing an in-place
/// DCT (Discrete Cosine Transform) on a data slice.
pub trait PxdctExecutor<T> {
    /// Executes the DCT transform in-place on the given data buffer.
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError>;
    /// Returns the length of the transform supported by this executor.
    fn length(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct2_roundtrip() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f32(array.len()).unwrap();

            dct_forward.execute(&mut working_array).unwrap();
            dct_inverse.execute(&mut working_array).unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {i} for size {i}");
            });
        }
    }
}
