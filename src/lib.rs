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
mod butterflies;
mod dct2;
mod dct3;
mod dst2;
mod dst3;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod pxdct_error;
mod spectrum_mul;
mod split_radix;
mod twiddles;
mod util;

use crate::butterflies::{
    Dct2Butterfly2, Dct2Butterfly3, Dct2Butterfly4, Dct2Butterfly8, Dct2Butterfly16,
    Dst2Butterfly2, Dst2Butterfly4,
};
use crate::dct2::Dct2Fft;
use crate::dct3::Dct3Fft;
use crate::dst2::Dst2Fft;
use crate::dst3::Dst3Fft;
use crate::split_radix::SplitRadixDct2;
pub use pxdct_error::PxdctError;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// The main entry point for creating DCT (Discrete Cosine Transform) executors.
///
/// `Pxdct` provides convenient factory methods to construct optimized
/// executors for DCT-II and DCT-III transforms using single (`f32`) or
/// double (`f64`) precision. Each executor implements the [`PxdctExecutor`]
/// trait and can be used to perform an in-place DCT transform on a data slice.
pub struct Pxdct {}

macro_rules! make_dct2_butterflies {
    ($length: expr, $f_type: ident) => {
        if $length == 2 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dct2Butterfly2::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if $length == 3 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dct2Butterfly3::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if $length == 4 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dct2Butterfly4::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if $length == 8 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dct2Butterfly8::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if $length == 16 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dct2Butterfly16::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        }
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
macro_rules! make_avx_dct2_butterflies {
    ($length: expr, $f_type: ident) => {
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")
        {
            use crate::avx::{
                AvxDct2Butterfly3, AvxDct2Butterfly4, AvxDct2Butterfly8, AvxDct2Butterfly16,
            };
            if $length == 2 {
                static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(Dct2Butterfly2::default())
                            as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                    })
                    .clone());
            } else if $length == 3 {
                static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDct2Butterfly3::default())
                            as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                    })
                    .clone());
            } else if $length == 4 {
                static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDct2Butterfly4::default())
                            as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                    })
                    .clone());
            } else if $length == 8 {
                static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDct2Butterfly8::default())
                            as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                    })
                    .clone());
            } else if $length == 16 {
                static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDct2Butterfly16::default())
                            as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                    })
                    .clone());
            }
        }
    };
}

static DCT2_SPLIT_RADIX_CACHE_F32: OnceLock<
    RwLock<HashMap<usize, Arc<dyn PxdctExecutor<f32> + Send + Sync>>>,
> = OnceLock::new();
static DCT2_SPLIT_RADIX_CACHE_F64: OnceLock<
    RwLock<HashMap<usize, Arc<dyn PxdctExecutor<f64> + Send + Sync>>>,
> = OnceLock::new();

impl Pxdct {
    /// Creates a single-precision (f32) DCT-II executor.
    pub fn make_dct2_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            make_avx_dct2_butterflies!(length, f32);
        }
        make_dct2_butterflies!(length, f32);

        if length.is_power_of_two() && length > 2 {
            return if length < 16384 {
                let rw_lock =
                    DCT2_SPLIT_RADIX_CACHE_F32.get_or_init(|| RwLock::new(HashMap::new()));
                match rw_lock.write() {
                    Ok(mut v) => {
                        if let Some(a) = v.get(&length) {
                            return Ok(a.clone());
                        }
                        let new_arc = Arc::new(SplitRadixDct2::new(
                            length,
                            Pxdct::make_dct2_f32(length / 2)?,
                            Pxdct::make_dct2_f32(length / 4)?,
                        )?);
                        v.insert(length, new_arc.clone());
                        Ok(new_arc)
                    }
                    Err(_) => Ok(Arc::new(SplitRadixDct2::new(
                        length,
                        Pxdct::make_dct2_f32(length / 2)?,
                        Pxdct::make_dct2_f32(length / 4)?,
                    )?)),
                }
            } else {
                Ok(Arc::new(SplitRadixDct2::new(
                    length,
                    Pxdct::make_dct2_f32(length / 2)?,
                    Pxdct::make_dct2_f32(length / 4)?,
                )?))
            };
        }

        Dct2Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DCT-II executor.
    pub fn make_dct2_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            make_avx_dct2_butterflies!(length, f64);
        }
        make_dct2_butterflies!(length, f64);

        if length.is_power_of_two() && length > 2 {
            return if length < 16384 {
                let rw_lock =
                    DCT2_SPLIT_RADIX_CACHE_F64.get_or_init(|| RwLock::new(HashMap::new()));
                match rw_lock.write() {
                    Ok(mut v) => {
                        if let Some(a) = v.get(&length) {
                            return Ok(a.clone());
                        }
                        let new_arc = Arc::new(SplitRadixDct2::new(
                            length,
                            Pxdct::make_dct2_f64(length / 2)?,
                            Pxdct::make_dct2_f64(length / 4)?,
                        )?);
                        v.insert(length, new_arc.clone());
                        Ok(new_arc)
                    }
                    Err(_) => Ok(Arc::new(SplitRadixDct2::new(
                        length,
                        Pxdct::make_dct2_f64(length / 2)?,
                        Pxdct::make_dct2_f64(length / 4)?,
                    )?)),
                }
            } else {
                Ok(Arc::new(SplitRadixDct2::new(
                    length,
                    Pxdct::make_dct2_f64(length / 2)?,
                    Pxdct::make_dct2_f64(length / 4)?,
                )?))
            };
        }

        Dct2Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
    }

    /// Creates a single-precision (f32) DCT-III executor.
    pub fn make_dct3_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        Dct3Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DCT-III executor.
    pub fn make_dct3_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        Dct3Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
    }

    /// Creates a single-precision (f32) DST-II executor.
    pub fn make_dst2_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")
        {
            #[allow(clippy::collapsible_if)]
            if length == 4 {
                use crate::avx::AvxDst2Butterfly4;
                static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDst2Butterfly4::default())
                            as Arc<dyn PxdctExecutor<f32> + Send + Sync>
                    })
                    .clone());
            }
        }
        if length == 2 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst2Butterfly2::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
                })
                .clone());
        } else if length == 4 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst2Butterfly4::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
                })
                .clone());
        }
        Dst2Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DST-II executor.
    pub fn make_dst2_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")
        {
            #[allow(clippy::collapsible_if)]
            if length == 4 {
                use crate::avx::AvxDst2Butterfly4;
                static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
                return Ok(Q
                    .get_or_init(|| {
                        Arc::new(AvxDst2Butterfly4::default())
                            as Arc<dyn PxdctExecutor<f64> + Send + Sync>
                    })
                    .clone());
            }
        }
        if length == 2 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst2Butterfly2::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
                })
                .clone());
        } else if length == 4 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst2Butterfly4::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
                })
                .clone());
        }
        Dst2Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
    }

    /// Creates a single-precision (f32) DST-III executor.
    pub fn make_dst3_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        Dst3Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f32> + Send + Sync>)
    }

    /// Creates a double-precision (f64) DST-III executor.
    pub fn make_dst3_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        Dst3Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
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

    pub(crate) fn naive_dct2(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f64) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }

        result
    }

    pub(crate) fn naive_dst2(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f64 + 1.0) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }
        result
    }

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

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst2_roundtrip() {
        for i in 4..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dst3_f32(array.len()).unwrap();

            dct_forward.execute(&mut working_array).unwrap();
            dct_inverse.execute(&mut working_array).unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }
}
