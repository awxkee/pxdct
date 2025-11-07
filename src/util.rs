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
use crate::PxdctError;
use crate::twiddles::FftTrigonometry;
use num_traits::{Float, MulAdd};
use std::fmt::Debug;
use std::ops::{Add, Mul};
use zaft::{FftDirection, FftExecutor, Zaft};

pub(crate) trait DctSample:
    FftTrigonometry
    + Float
    + Copy
    + 'static
    + Clone
    + Default
    + FftProvider<Self>
    + Debug
    + MulAdd<Self, Output = Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + FftSpectrumMulFactory<Self>
    + Half
{
}

impl DctSample for f32 {}

impl DctSample for f64 {}

pub(crate) trait Half {
    const HALF: Self;
}

impl Half for f32 {
    const HALF: Self = 0.5;
}

impl Half for f64 {
    const HALF: Self = 0.5;
}

pub(crate) trait FftProvider<T> {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, PxdctError>;
}

impl FftProvider<f32> for f32 {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, PxdctError> {
        match direction {
            FftDirection::Forward => Zaft::make_forward_fft_f32(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
            FftDirection::Inverse => Zaft::make_inverse_fft_f32(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
        }
    }
}

impl FftProvider<f64> for f64 {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, PxdctError> {
        match direction {
            FftDirection::Forward => Zaft::make_forward_fft_f64(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
            FftDirection::Inverse => Zaft::make_inverse_fft_f64(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
        }
    }
}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::PxdctError::OutOfMemory($n))?;
        v.resize($n, $elem);
        v
    }};
}

use crate::spectrum_mul::FftSpectrumMulFactory;
pub(crate) use try_vec;

#[inline]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(crate) fn has_valid_avx() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}
