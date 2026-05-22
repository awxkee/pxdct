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
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod bidirectional;
mod butterflies;
mod dst3;
mod dst3_butterfly;
mod factory_dct1;
mod factory_dct2;
mod factory_dct3;
mod factory_dct4;
mod factory_dct7;
mod factory_dst2;
mod factory_dst7;
mod factory_scaled_dct2;
mod identity;
mod mdct;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod prime_factors;
mod pxdct_error;
mod scaling;
mod spectrum_mul;
mod transpose;
mod twiddles;
mod two_dims;
mod type1;
mod type2;
mod type3;
mod type4;
mod type5;
mod type6;
mod type7;
mod type8;
mod util;

use crate::dst3::Dst3Fft;
use crate::dst3_butterfly::{
    Dst3Butterfly2, Dst3Butterfly3, Dst3Butterfly4, Dst3Butterfly5, Dst3Butterfly8, Dst3Butterfly16,
};
use crate::factory_dct1::Dct1Factory;
use crate::factory_dct2::Dct2Factory;
use crate::factory_dct3::Dct3Factory;
use crate::factory_dct4::Dct4Factory;
use crate::factory_dct7::Dct7Factory;
use crate::factory_dst2::Dst2Factory;
use crate::factory_dst7::Dst7Factory;
use crate::factory_scaled_dct2::ScaledDct2Factory;
use crate::identity::DctIdentity;
use crate::prime_factors::PrimeFactors;
use crate::scaling::wrap_with_scaling;
use crate::transpose::TransposeFactory;
use crate::two_dims::TwoDimensionalDct;
use crate::type2::Dct2Fft;
use crate::type3::SplitRadixDst3;
use crate::util::DctSample;
use num_traits::AsPrimitive;
pub use pxdct_error::PxdctError;
pub use scaling::{Scaling, TransformKind};
use std::sync::{Arc, OnceLock};
pub use two_dims::MultidimensionalDctExecutor;
use zaft::FftDirection;

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
            return Ok($f_type::dct2_butterfly2());
        } else if $length == 3 {
            return Ok($f_type::dct2_butterfly3());
        } else if $length == 4 {
            return Ok($f_type::dct2_butterfly4());
        } else if $length == 5 {
            return Ok($f_type::dct2_butterfly5());
        } else if $length == 6 {
            return Ok($f_type::dct2_butterfly6());
        } else if $length == 7 {
            return Ok($f_type::dct2_butterfly7());
        } else if $length == 8 {
            return Ok($f_type::dct2_butterfly8());
        } else if $length == 9 {
            return Ok($f_type::dct2_butterfly9());
        } else if $length == 10 {
            return Ok($f_type::dct2_butterfly10());
        } else if $length == 11 {
            return Ok($f_type::dct2_butterfly11());
        } else if $length == 12 {
            return Ok($f_type::dct2_butterfly12());
        } else if $length == 13 {
            return Ok($f_type::dct2_butterfly13());
        } else if $length == 14 {
            return Ok($f_type::dct2_butterfly14());
        } else if $length == 15 {
            return Ok($f_type::dct2_butterfly15());
        } else if $length == 16 {
            return Ok($f_type::dct2_butterfly16());
        } else if $length == 17 {
            return Ok($f_type::dct2_butterfly17());
        } else if $length == 18 {
            return Ok($f_type::dct2_butterfly18());
        } else if $length == 19 {
            return Ok($f_type::dct2_butterfly19());
        } else if $length == 20 {
            return Ok($f_type::dct2_butterfly20());
        } else if $length == 21 {
            return Ok($f_type::dct2_butterfly21());
        } else if $length == 22 {
            return Ok($f_type::dct2_butterfly22());
        } else if $length == 23 {
            return Ok($f_type::dct2_butterfly23());
        } else if $length == 24 {
            return Ok($f_type::dct2_butterfly24());
        } else if $length == 25 {
            return Ok($f_type::dct2_butterfly25());
        } else if $length == 26 {
            return Ok($f_type::dct2_butterfly26());
        } else if $length == 27 {
            return Ok($f_type::dct2_butterfly27());
        } else if $length == 29 {
            return Ok($f_type::dct2_butterfly29());
        } else if $length == 30 {
            return Ok($f_type::dct2_butterfly30());
        } else if $length == 31 {
            return Ok($f_type::dct2_butterfly31());
        } else if $length == 32 {
            return Ok($f_type::dct2_butterfly32());
        } else if $length == 35 {
            return Ok($f_type::dct2_butterfly35());
        } else if $length == 36 {
            return Ok($f_type::dct2_butterfly36());
        } else if $length == 37 {
            return Ok($f_type::dct2_butterfly37());
        } else if $length == 42 {
            return Ok($f_type::dct2_butterfly42());
        } else if $length == 48 {
            return Ok($f_type::dct2_butterfly48());
        } else if $length == 49 {
            return Ok($f_type::dct2_butterfly49());
        } else if $length == 64 {
            return Ok($f_type::dct2_butterfly64());
        } else if $length == 81 {
            return Ok($f_type::dct2_butterfly81());
        } else if $length == 128 {
            return Ok($f_type::dct2_butterfly128());
        } else if $length == 216 {
            return Ok($f_type::dct2_butterfly216());
        } else if $length == 243 {
            return Ok($f_type::dct2_butterfly243());
        } else if $length == 256 {
            return Ok($f_type::dct2_butterfly256());
        } else if $length == 512 {
            return Ok($f_type::dct2_butterfly512());
        }
    };
}

macro_rules! generate_dst3_butterflies {
    ($length: expr, $f_type: ident) => {
        let length = $length;
        if length == 2 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly2::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if length == 3 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly3::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if length == 4 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly4::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if length == 5 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly5::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if length == 8 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly8::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        } else if length == 16 {
            static Q: OnceLock<Arc<dyn PxdctExecutor<$f_type> + Send + Sync>> = OnceLock::new();
            return Ok(Q
                .get_or_init(|| {
                    Arc::new(Dst3Butterfly16::default())
                        as Arc<dyn PxdctExecutor<$f_type> + Send + Sync>
                })
                .clone());
        }
    };
}

impl Pxdct {
    fn dct2_strategy<T: DctSample + Dct2Factory + Send + Sync>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if length == 1 {
            return Ok(Arc::new(DctIdentity::default()));
        }

        make_dct2_butterflies!(length, T);

        if length.is_power_of_two() && length > 2 {
            return T::split_radix(
                length,
                Pxdct::dct2_strategy(length / 2)?,
                Pxdct::dct2_strategy(length / 4)?,
            );
        }

        if length.is_multiple_of(6) {
            return T::mixed_radix6(length, Pxdct::dct2_strategy(length / 6)?);
        }

        if length.is_multiple_of(9) {
            return T::mixed_radix9(length, Pxdct::dct2_strategy(length / 9)?);
        }

        if length.is_multiple_of(3) {
            return T::mixed_radix3(length, Pxdct::dct2_strategy(length / 3)?);
        }

        if length.is_multiple_of(5) {
            return T::mixed_radix5(length, Pxdct::dct2_strategy(length / 5)?);
        }

        if length.is_multiple_of(7) {
            return T::mixed_radix7(length, Pxdct::dct2_strategy(length / 7)?);
        }

        if length.is_multiple_of(11) {
            return T::mixed_radix11(length, Pxdct::dct2_strategy(length / 11)?);
        }

        if length.is_multiple_of(13) {
            return T::mixed_radix13(length, Pxdct::dct2_strategy(length / 13)?);
        }

        let prime_factors = PrimeFactors::from_number(length as u64);

        // if number is too big and 2^k*Q is big this is not super effective to use mixed-radix,
        // go with FFT instead
        let factor2 = prime_factors.factor_of_2();

        if (length.is_multiple_of(2) && factor2 < 5) || (length.is_multiple_of(2) && length < 1000)
        {
            return T::mixed_radix2(length, Pxdct::dct2_strategy(length / 2)?);
        }

        if length == 29 * 11 {
            return T::dct2_relatively_prime(Pxdct::dct2_strategy(29)?, Pxdct::dct2_strategy(11)?);
        }

        Dct2Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<T> + Send + Sync>)
    }

    fn strategy_scaled_dct2<T: DctSample + ScaledDct2Factory + Dct2Factory>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if length == 1 {
            return Ok(Arc::new(DctIdentity::default()));
        }
        if length.is_power_of_two() {
            return match length {
                2 => Ok(T::scaled_dct2_butterfly2()),
                4 => Ok(T::scaled_dct2_butterfly4()),
                8 => Ok(T::scaled_dct2_butterfly8()),
                16 => Ok(T::scaled_dct2_butterfly16()),
                32 => Ok(T::scaled_dct2_butterfly32()),
                64 => Ok(T::scaled_dct2_butterfly64()),
                128 => Ok(T::scaled_dct2_butterfly128()),
                256 => Ok(T::scaled_dct2_butterfly256()),
                512 => Ok(T::scaled_dct2_butterfly512()),
                _ => T::scaled_split_radix(
                    length,
                    Pxdct::dct2_strategy(length / 2)?,
                    Pxdct::dct2_strategy(length / 4)?,
                ),
            };
        }
        wrap_with_scaling(
            Pxdct::dct2_strategy(length)?,
            TransformKind::Dct2,
            Scaling::Scale,
        )
    }

    /// Creates a single-precision (f32) DCT-II executor.
    pub fn make_dct2_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dct2_strategy(length)
    }

    /// Creates a scaled single-precision (f32) DCT-II executor.
    ///
    /// Results are scaled by SQRT(2/N).
    /// Scaling routine is completely incorporated only in power of 2 executors.
    /// For anything else if absolute performance is preferred consider to
    /// incorporate scaling factor into your processing routine.
    pub fn make_scaled_dct2_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::strategy_scaled_dct2(length)
    }

    /// Creates a double-precision (f64) DCT-II executor.
    pub fn make_dct2_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dct2_strategy(length)
    }

    /// Creates a scaled double-precision (f64) DCT-II executor.
    ///
    /// Results are scaled by SQRT(2/N).
    /// Scaling routine is completely incorporated only in power of 2 executors.
    /// For anything else if absolute performance is preferred consider to
    /// incorporate scaling factor into your processing routine.
    pub fn make_scaled_dct2_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::strategy_scaled_dct2(length)
    }

    fn dct3_strategy<T: Copy + Dct3Factory>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        match length {
            1 => return Ok(T::dct3_identity()),
            2 => return Ok(T::dct3_butterfly2()),
            3 => return Ok(T::dct3_butterfly3()),
            4 => return Ok(T::dct3_butterfly4()),
            5 => return Ok(T::dct3_butterfly5()),
            6 => return Ok(T::dct3_butterfly6()),
            7 => return Ok(T::dct3_butterfly7()),
            8 => return Ok(T::dct3_butterfly8()),
            9 => return Ok(T::dct3_butterfly9()),
            10 => return Ok(T::dct3_butterfly10()),
            11 => return Ok(T::dct3_butterfly11()),
            12 => return Ok(T::dct3_butterfly12()),
            13 => return Ok(T::dct3_butterfly13()),
            14 => return Ok(T::dct3_butterfly14()),
            15 => return Ok(T::dct3_butterfly15()),
            16 => return Ok(T::dct3_butterfly16()),
            18 => return Ok(T::dct3_butterfly18()),
            20 => return Ok(T::dct3_butterfly20()),
            21 => return Ok(T::dct3_butterfly21()),
            24 => return Ok(T::dct3_butterfly24()),
            26 => return Ok(T::dct3_butterfly26()),
            28 => return Ok(T::dct3_butterfly28()),
            30 => return Ok(T::dct3_butterfly30()),
            32 => return Ok(T::dct3_butterfly32()),
            35 => return Ok(T::dct3_butterfly35()),
            36 => return Ok(T::dct3_butterfly36()),
            64 => return Ok(T::dct3_butterfly64()),
            _ => {}
        }

        if length.is_multiple_of(9) {
            return T::dct3_mixed_radix9(Self::dct3_strategy(length / 9)?);
        }
        if length.is_multiple_of(7) {
            return T::dct3_mixed_radix7(Self::dct3_strategy(length / 7)?);
        }
        if length.is_multiple_of(5) {
            return T::dct3_mixed_radix5(Self::dct3_strategy(length / 5)?);
        }
        if length.is_multiple_of(3) {
            return T::dct3_mixed_radix3(Self::dct3_strategy(length / 3)?);
        }

        if length.is_power_of_two() && length > 2 {
            return T::dct3_split_radix(
                length,
                Self::dct3_strategy(length / 2)?,
                Self::dct3_strategy(length / 4)?,
            );
        }

        T::dct3_fft(length)
    }

    fn dct1_strategy<T: Copy + Dct1Factory + Dct3Factory>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError> {
        if length < 2 {
            return Err(PxdctError::MinimumPoints(2, "DCT-I".to_string()));
        }

        match length {
            2 => return Ok(T::dct1_butterfly2()),
            3 => return Ok(T::dct1_butterfly3()),
            4 => return Ok(T::dct1_butterfly4()),
            5 => return Ok(T::dct1_butterfly5()),
            6 => return Ok(T::dct1_butterfly6()),
            7 => return Ok(T::dct1_butterfly7()),
            8 => return Ok(T::dct1_butterfly8()),
            9 => return Ok(T::dct1_butterfly9()),
            17 => return Ok(T::dct1_butterfly17()),
            _ => {}
        }

        let n = length - 1;
        if n > 0 && n.is_multiple_of(2) {
            let n1 = n / 2;

            let half_p1_dct1 = Pxdct::dct1_strategy(n1 + 1)?;
            let half_dct3 = Pxdct::dct3_strategy(n1)?;

            return T::split_radix(length, half_p1_dct1, half_dct3);
        }

        T::dct1_fft(length)
    }

    /// Creates a single-precision (f32) DCT-I executor.
    pub fn make_dct1_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dct1_strategy(length)
    }

    /// Creates a single-precision (f64) DCT-I executor.
    pub fn make_dct1_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dct1_strategy(length)
    }

    /// Creates a single-precision (f32) DST-I executor.
    pub fn make_dst1_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        use crate::type1::Dst1Fft;
        Ok(Arc::new(Dst1Fft::new(length)?))
    }

    /// Creates a single-precision (f64) DST-I executor.
    pub fn make_dst1_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        use crate::type1::Dst1Fft;
        Ok(Arc::new(Dst1Fft::new(length)?))
    }

    /// Creates a single-precision (f32) DCT-III executor.
    pub fn make_dct3_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dct3_strategy(length)
    }

    /// Creates a single-precision (f32) DCT-III executor.
    /// Results are scaled by SQRT(2/N).
    /// Scaling routine is completely incorporated only in power of 2 executors.
    /// For anything else if absolute performance is preferred consider to
    /// incorporate scaling factor into your processing routine.
    pub fn make_scaled_dct3_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        wrap_with_scaling(
            Pxdct::dct3_strategy(length)?,
            TransformKind::Dct3,
            Scaling::Scale,
        )
    }

    /// Creates a double-precision (f64) DCT-III executor.
    pub fn make_dct3_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dct3_strategy(length)
    }

    /// Creates a double-precision (f64) DCT-III executor.
    pub fn make_scaling_dct3_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        wrap_with_scaling(
            Pxdct::dct3_strategy(length)?,
            TransformKind::Dct3,
            Scaling::Scale,
        )
    }

    fn dst2_strategy<T: Copy + Dst2Factory + Dct2Factory + DctSample>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        match length {
            2 => return Ok(T::dst2_butterfly2()),
            3 => return Ok(T::dst2_butterfly3()),
            4 => return Ok(T::dst2_butterfly4()),
            5 => return Ok(T::dst2_butterfly5()),
            6 => return Ok(T::dst2_butterfly6()),
            7 => return Ok(T::dst2_butterfly7()),
            8 => return Ok(T::dst2_butterfly8()),
            9 => return Ok(T::dst2_butterfly9()),
            16 => return Ok(T::dst2_butterfly16()),
            _ => {}
        }

        if length.is_power_of_two() && length > 2 {
            return T::dst2_split_radix(
                length,
                Pxdct::dct2_strategy(length / 2)?,
                Pxdct::dct2_strategy(length / 4)?,
            );
        }

        // this is not very performant at the moment, much easier go with FFT for many cases
        // but it could be improved so we'll just keep it for simple cases
        if length <= 24 && length.is_multiple_of(3) {
            return T::dst2_mixed_radix3(Pxdct::dst2_strategy(length / 3)?);
        }

        T::dst2_fft(length)
    }

    /// Creates a single-precision (f32) DST-II executor.
    pub fn make_dst2_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dst2_strategy(length)
    }

    /// Creates a double-precision (f64) DST-II executor.
    pub fn make_dst2_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dst2_strategy(length)
    }

    /// Creates a single-precision (f32) DST-III executor.
    pub fn make_dst3_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        generate_dst3_butterflies!(length, f32);

        if length.is_power_of_two() && length > 2 {
            return Ok(Arc::new(SplitRadixDst3::new(
                length,
                Pxdct::make_dct3_f32(length / 2)?,
                Pxdct::make_dct3_f32(length / 4)?,
            )?));
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

        generate_dst3_butterflies!(length, f64);

        if length.is_power_of_two() && length > 2 {
            return Ok(Arc::new(SplitRadixDst3::new(
                length,
                Pxdct::make_dct3_f64(length / 2)?,
                Pxdct::make_dct3_f64(length / 4)?,
            )?));
        }

        Dst3Fft::new(length).map(|x| Arc::new(x) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
    }

    fn strategy_dct4<T: DctSample + Dct4Factory + Dct2Factory>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        if length == 1 {
            return Ok(T::dct4_identity());
        } else if length == 2 {
            return Ok(T::dct4_butterfly2());
        } else if length == 3 {
            return Ok(T::dct4_butterfly3());
        } else if length == 4 {
            return Ok(T::dct4_butterfly4());
        } else if length == 5 {
            return Ok(T::dct4_butterfly5());
        } else if length == 6 {
            return Ok(T::dct4_butterfly6());
        } else if length == 7 {
            return Ok(T::dct4_butterfly7());
        } else if length == 8 {
            return Ok(T::dct4_butterfly8());
        } else if length == 9 {
            return Ok(T::dct4_butterfly9());
        } else if length == 10 {
            return Ok(T::dct4_butterfly10());
        } else if length == 11 {
            return Ok(T::dct4_butterfly11());
        } else if length == 12 {
            return Ok(T::dct4_butterfly12());
        } else if length == 13 {
            return Ok(T::dct4_butterfly13());
        } else if length == 14 {
            return Ok(T::dct4_butterfly14());
        } else if length == 16 {
            return Ok(T::dct4_butterfly16());
        } else if length == 17 {
            return Ok(T::dct4_butterfly17());
        } else if length == 18 {
            return Ok(T::dct4_butterfly18());
        } else if length == 19 {
            return Ok(T::dct4_butterfly19());
        } else if length == 20 {
            return Ok(T::dct4_butterfly20());
        } else if length == 22 {
            return Ok(T::dct4_butterfly22());
        } else if length == 23 {
            return Ok(T::dct4_butterfly23());
        } else if length == 24 {
            return Ok(T::dct4_butterfly24());
        } else if length == 26 {
            return Ok(T::dct4_butterfly26());
        } else if length == 27 {
            return Ok(T::dct4_butterfly27());
        } else if length == 28 {
            return Ok(T::dct4_butterfly28());
        } else if length == 29 {
            return Ok(T::dct4_butterfly29());
        } else if length == 30 {
            return Ok(T::dct4_butterfly30());
        } else if length == 32 {
            return Ok(T::dct4_butterfly32());
        }

        if length.is_multiple_of(9) {
            let half_length = length / 9;
            return T::dct4_mixed_radix9(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(7) {
            let half_length = length / 7;
            return T::dct4_mixed_radix7(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(5) {
            let half_length = length / 5;
            return T::dct4_mixed_radix5(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(11) {
            let half_length = length / 11;
            return T::dct4_mixed_radix11(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(13) {
            let half_length = length / 13;
            return T::dct4_mixed_radix13(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(17) {
            let half_length = length / 17;
            return T::dct4_mixed_radix17(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(19) {
            let half_length = length / 19;
            return T::dct4_mixed_radix19(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(3) {
            let half_length = length / 3;
            return T::dct4_mixed_radix3(length, Pxdct::strategy_dct4(half_length)?);
        }

        if length.is_multiple_of(2) {
            if length.is_power_of_two() {
                return T::dct4_radix2(length, Pxdct::dct2_strategy(length / 2)?);
            }

            let half_length = length / 2;
            // If more than one factor of 2 remains, go straight to FFT-based path
            // rather than recursing through mixed-radix-2 repeatedly
            if half_length.is_multiple_of(2) {
                return T::dct4_fft_even(
                    T::make_fft(length, FftDirection::Forward)
                        .map_err(|x| PxdctError::FftError(x.to_string()))?,
                );
            }
            return T::dct4_mixed_radix2(length, Pxdct::dct2_strategy(half_length)?);
        }

        T::dct4_fft_odd(T::make_fft_r2c(length).map_err(|x| PxdctError::FftError(x.to_string()))?)
    }

    /// Creates a single-precision (f32) DCT-IV executor.
    pub fn make_dct4_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::strategy_dct4(length)
    }

    /// Creates a scaled single-precision (f32) DCT-IV executor.
    pub fn make_scaled_dct4_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        wrap_with_scaling(
            Pxdct::strategy_dct4(length)?,
            TransformKind::Dct4,
            Scaling::Scale,
        )
    }

    /// Creates a double-precision (f32) DCT-IV executor.
    pub fn make_dct4_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::strategy_dct4(length)
    }

    /// Creates a scaled single-precision (f32) DCT-IV executor.
    pub fn make_scaled_dct4_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        wrap_with_scaling(
            Pxdct::strategy_dct4(length)?,
            TransformKind::Dct4,
            Scaling::Scale,
        )
    }

    /// Creates a single-precision (f32) DST-IV executor.
    pub fn make_dst4_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        use crate::type4::Dst4OverDct4;
        Ok(Arc::new(Dst4OverDct4::new(Pxdct::strategy_dct4(length)?)?))
    }

    /// Creates a double-precision (f64) DST-IV executor.
    pub fn make_dst4_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        use crate::type4::Dst4OverDct4;
        Ok(Arc::new(Dst4OverDct4::new(Pxdct::strategy_dct4(length)?)?))
    }

    /// Creates a single-precision (f32) DCT-V executor.
    pub fn make_dct5_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type5::Dct5Fft;
        Ok(Arc::new(Dct5Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DCT-V executor.
    pub fn make_dct5_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type5::Dct5Fft;
        Ok(Arc::new(Dct5Fft::new(length)?))
    }

    /// Creates a single-precision (f32) DST-V executor.
    pub fn make_dst5_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type5::Dst5Fft;
        Ok(Arc::new(Dst5Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DST-V executor.
    pub fn make_dst5_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type5::Dst5Fft;
        Ok(Arc::new(Dst5Fft::new(length)?))
    }

    /// Creates a single-precision (f32) DCT-VI executor.
    pub fn make_dct6_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type6::Dct6Fft;
        Ok(Arc::new(Dct6Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DCT-VI executor.
    pub fn make_dct6_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type6::Dct6Fft;
        Ok(Arc::new(Dct6Fft::new(length)?))
    }

    /// Creates a single-precision (f32) DST-VI executor.
    pub fn make_dst6_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type6::Dst6Fft;
        Ok(Arc::new(Dst6Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DST-VI executor.
    pub fn make_dst6_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type6::Dst6Fft;
        Ok(Arc::new(Dst6Fft::new(length)?))
    }

    fn dst7_strategy<T: Copy + Dst7Factory + DctSample>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        match length {
            2 => return Ok(T::dst7_butterfly2()),
            3 => return Ok(T::dst7_butterfly3()),
            4 => return Ok(T::dst7_butterfly4()),
            5 => return Ok(T::dst7_butterfly5()),
            6 => return Ok(T::dst7_butterfly6()),
            7 => return Ok(T::dst7_butterfly7()),
            8 => return Ok(T::dst7_butterfly8()),
            16 => return Ok(T::dst7_butterfly16()),
            _ => {}
        }

        T::dst7_fft(length)
    }

    /// Creates a single-precision (f32) DST-VII executor.
    pub fn make_dst7_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dst7_strategy(length)
    }

    /// Creates a double-precision (f64) DST-VII executor.
    pub fn make_dst7_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dst7_strategy(length)
    }

    fn dct7_strategy<T: Copy + Dct7Factory + DctSample>(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        match length {
            2 => return Ok(T::dct7_butterfly2()),
            3 => return Ok(T::dct7_butterfly3()),
            4 => return Ok(T::dct7_butterfly4()),
            8 => return Ok(T::dct7_butterfly8()),
            _ => {}
        }

        T::dct7_fft(length)
    }

    /// Creates a single-precision (f32) DCT-VII executor.
    pub fn make_dct7_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        Pxdct::dct7_strategy(length)
    }

    /// Creates a double-precision (f64) DCT-VII executor.
    pub fn make_dct7_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        Pxdct::dct7_strategy(length)
    }

    /// Creates a single-precision (f32) DST-VIII executor.
    pub fn make_dst8_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type8::Dst8Fft;
        Ok(Arc::new(Dst8Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DST-VIII executor.
    pub fn make_dst8_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type8::Dst8Fft;
        Ok(Arc::new(Dst8Fft::new(length)?))
    }

    /// Creates a single-precision (f32) DST-VIII executor.
    pub fn make_dct8_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type8::Dct8Fft;
        Ok(Arc::new(Dct8Fft::new(length)?))
    }

    /// Creates a double-precision (f64) DCT-VIII executor.
    pub fn make_dct8_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        use crate::type8::Dct8Fft;
        Ok(Arc::new(Dct8Fft::new(length)?))
    }

    /// Creates a single-precision (f32) MDCT executor.
    pub fn make_mdct_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if !length.is_multiple_of(2) {
            return Err(PxdctError::OnlyEvenTransform(length));
        }
        use crate::mdct::MdctFft;
        Ok(Arc::new(MdctFft::new(length)?))
    }

    /// Creates a double-precision (f64) MDCT executor.
    pub fn make_mdct_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if !length.is_multiple_of(2) {
            return Err(PxdctError::OnlyEvenTransform(length));
        }
        use crate::mdct::MdctFft;
        Ok(Arc::new(MdctFft::new(length)?))
    }

    /// Creates a single-precision (f32) IMDCT executor.
    pub fn make_imdct_f32(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if !length.is_multiple_of(2) {
            return Err(PxdctError::OnlyEvenTransform(length));
        }
        use crate::mdct::ImdctFft;
        Ok(Arc::new(ImdctFft::new(length)?))
    }

    /// Creates a double-precision (f64) IMDCT executor.
    pub fn make_imdct_f64(
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }
        if !length.is_multiple_of(2) {
            return Err(PxdctError::OnlyEvenTransform(length));
        }
        use crate::mdct::ImdctFft;
        Ok(Arc::new(ImdctFft::new(length)?))
    }

    /// Creates 2D DCT executor.
    ///
    /// For matrix WxH to get an inverse use H as width and W as height.
    pub fn make_2d_dct_f32(
        width_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
        height_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Arc<dyn MultidimensionalDctExecutor<f32> + Send + Sync> {
        let width = width_dct.length();
        let height = height_dct.length();
        let width_scratch_size = width_dct.scratch_size();
        let height_scratch_size = height_dct.scratch_size();
        Arc::new(TwoDimensionalDct {
            width,
            height,
            height_executor: height_dct,
            width_executor: width_dct,
            transpose_width_to_height: f32::make_transpose(width, height),
            width_scratch_size,
            height_scratch_size,
        })
    }

    /// Creates 2D DCT executor.
    ///
    /// For matrix WxH to get an inverse use H as width and W as height.
    pub fn make_2d_dct_f64(
        width_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
        height_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Arc<dyn MultidimensionalDctExecutor<f64> + Send + Sync> {
        let width = width_dct.length();
        let height = height_dct.length();
        let width_scratch_size = width_dct.scratch_size();
        let height_scratch_size = height_dct.scratch_size();
        Arc::new(TwoDimensionalDct {
            width,
            height,
            height_executor: height_dct,
            width_executor: width_dct,
            transpose_width_to_height: f64::make_transpose(width, height),
            width_scratch_size,
            height_scratch_size,
        })
    }

    /// Creates a single-precision (f32) executor for any supported transform type
    /// with the requested normalization.
    pub fn make_f32(
        kind: TransformKind,
        length: usize,
        scaling: Scaling,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        let inner = Self::make_raw_f32(kind, length)?;
        wrap_with_scaling(inner, kind, scaling)
    }

    /// Creates a double-precision (f64) executor for any supported transform type
    /// with the requested normalization.
    pub fn make_f64(
        kind: TransformKind,
        length: usize,
        scaling: Scaling,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        let inner = Self::make_raw_f64(kind, length)?;
        wrap_with_scaling(inner, kind, scaling)
    }

    fn make_raw_f32(
        kind: TransformKind,
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f32> + Send + Sync>, PxdctError> {
        match kind {
            TransformKind::Dct1 => Self::make_dct1_f32(length),
            TransformKind::Dct2 => Self::make_dct2_f32(length),
            TransformKind::Dct3 => Self::make_dct3_f32(length),
            TransformKind::Dct4 => Self::make_dct4_f32(length),
            TransformKind::Dct5 => Self::make_dct5_f32(length),
            TransformKind::Dct6 => Self::make_dct6_f32(length),
            TransformKind::Dct7 => Self::make_dct7_f32(length),
            TransformKind::Dct8 => Self::make_dct8_f32(length),
            TransformKind::Dst1 => Self::make_dst1_f32(length),
            TransformKind::Dst2 => Self::make_dst2_f32(length),
            TransformKind::Dst3 => Self::make_dst3_f32(length),
            TransformKind::Dst4 => Self::make_dst4_f32(length),
            TransformKind::Dst5 => Self::make_dst5_f32(length),
            TransformKind::Dst6 => Self::make_dst6_f32(length),
            TransformKind::Dst7 => Self::make_dst7_f32(length),
            TransformKind::Dst8 => Self::make_dst8_f32(length),
        }
    }

    fn make_raw_f64(
        kind: TransformKind,
        length: usize,
    ) -> Result<Arc<dyn PxdctExecutor<f64> + Send + Sync>, PxdctError> {
        match kind {
            TransformKind::Dct1 => Self::make_dct1_f64(length),
            TransformKind::Dct2 => Self::make_dct2_f64(length),
            TransformKind::Dct3 => Self::make_dct3_f64(length),
            TransformKind::Dct4 => Self::make_dct4_f64(length),
            TransformKind::Dct5 => Self::make_dct5_f64(length),
            TransformKind::Dct6 => Self::make_dct6_f64(length),
            TransformKind::Dct7 => Self::make_dct7_f64(length),
            TransformKind::Dct8 => Self::make_dct8_f64(length),
            TransformKind::Dst1 => Self::make_dst1_f64(length),
            TransformKind::Dst2 => Self::make_dst2_f64(length),
            TransformKind::Dst3 => Self::make_dst3_f64(length),
            TransformKind::Dst4 => Self::make_dst4_f64(length),
            TransformKind::Dst5 => Self::make_dst5_f64(length),
            TransformKind::Dst6 => Self::make_dst6_f64(length),
            TransformKind::Dst7 => Self::make_dst7_f64(length),
            TransformKind::Dst8 => Self::make_dst8_f64(length),
        }
    }
}

/// Trait implemented by all PXDCT executors.
///
/// This trait defines the common interface for performing an in-place
/// DCT (Discrete Cosine Transform) on a data slice.
pub trait PxdctExecutor<T> {
    /// Executes the DCT transform in-place on the given data buffer.
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError>;
    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError>;
    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError>;
    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError>;
    /// Returns the length of the transform supported by this executor.
    fn length(&self) -> usize;
    fn scratch_size(&self) -> usize;
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

    pub(crate) fn naive_scaled_dct2(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        let scale = (2f64 / input.len() as f64).sqrt();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f64) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry * scale);
        }

        result
    }

    #[allow(unused)]
    pub(crate) fn naive_scaled_dct2_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();

        let scale = (2f32 / input.len() as f32).sqrt();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f32) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry * scale);
        }

        result
    }
    #[allow(unused)]
    pub(crate) fn naive_dct2_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f32) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32);
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

    pub fn naive_dct3(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner =
                    (output_index as f64 + 0.5) * (input_index as f64) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }

        result
    }

    pub fn naive_dct3_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner =
                    (output_index as f32 + 0.5) * (input_index as f32) * std::f32::consts::PI
                        / (input.len() as f32);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }

        result
    }

    pub fn naive_dst3(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let sin_inner =
                    (output_index as f64 + 0.5) * (input_index as f64 + 1.0) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }
        result
    }

    pub fn naive_dct4(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }

        result
    }

    pub fn naive_dct4_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f32 + 0.5) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }

        result
    }

    pub(crate) fn naive_dst1(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f64 + 1.0) * (input_index as f64 + 1.0) * std::f64::consts::PI
                        / ((input.len() + 1) as f64);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dct1(input: &[f64]) -> Vec<f64> {
        let n = input.len() - 1; // N = len - 1
        let mut output = vec![0.0; input.len()];
        for k in 0..=n {
            let mut sum = input[0] + (if k % 2 == 0 { 1.0 } else { -1.0 }) * input[n];
            for j in 1..n {
                sum += 2.0
                    * input[j]
                    * (std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64)).cos();
            }
            output[k] = sum;
        }
        output
    }

    pub(crate) fn naive_dst6(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut result = Vec::with_capacity(n);
        for k in 0..n {
            let mut entry = 0.0;
            for i in 0..n {
                let sin_inner = (k as f64 + 1.0) * (2.0 * i as f64 + 1.0) * std::f64::consts::PI
                    / (2.0 * n as f64 + 1.0);
                entry += input[i] * sin_inner.sin();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst7(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f64 + 0.5) * (input_index as f64 + 1.0) * std::f64::consts::PI
                        / (input.len() as f64 + 0.5);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dct7(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner =
                    (output_index as f64 + 0.5) * (input_index as f64) * std::f64::consts::PI
                        / (input.len() as f64 - 0.5);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
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
    fn dct2_roundtrip_oof() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();

            let mut transient = vec![0f32; i];

            let dct_forward = Pxdct::make_dct2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f32(array.len()).unwrap();

            dct_forward
                .execute_into(&working_array, &mut transient)
                .expect(&format!("Failed to execute DCT on size {i}"));
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .expect(&format!("Failed to execute DCT on size {i}"));

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct2_roundtrip_f64() {
        for i in 1..250 {
            let mut array = vec![0f64; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct2_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f64(array.len()).unwrap();

            dct_forward.execute(&mut working_array).unwrap();
            dct_inverse.execute(&mut working_array).unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct2_roundtrip_oof_f64() {
        for i in 1..250 {
            let mut array = vec![0f64; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let mut transient = vec![0f64; i];

            let dct_forward = Pxdct::make_dct2_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f64(array.len()).unwrap();

            dct_forward
                .execute_into(&working_array, &mut transient)
                .expect(&format!("Failed to execute DCT on size {i}"));
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .expect(&format!("Failed to execute DCT on size {i}"));

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst2_roundtrip() {
        for i in 1..250 {
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

    #[test]
    fn dst2_roundtrip_oof() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();
            let mut transient = vec![0f32; i];

            let dct_forward = Pxdct::make_dst2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dst3_f32(array.len()).unwrap();

            dct_forward
                .execute_into(&working_array, &mut transient)
                .expect(&format!("Failed to execute DST-II on size {i}"));
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .expect(&format!("Failed to execute DST-III on size {i}"));

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct4_roundtrip() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct4_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f32(array.len()).unwrap();

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
    fn dct4_roundtrip_oof() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f32;
            }
            let mut working_array = array.clone();

            let mut transient = vec![0f32; i];

            let dct_forward = Pxdct::make_dct4_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f32(array.len()).unwrap();

            dct_forward
                .execute_into(&working_array, &mut transient)
                .expect(&format!("Failed to execute DCT-IV on size {i}"));
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .expect(&format!("Failed to execute DCT-IV on size {i}"));

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct4_roundtrip_f64() {
        for i in 1..300 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct4_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f64(array.len()).unwrap();

            dct_forward.execute(&mut working_array).unwrap();
            dct_inverse.execute(&mut working_array).unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.00001, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct4_roundtrip_oof_f64() {
        for i in 1..300 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();

            let mut transient = vec![0f64; i];

            let dct_forward = Pxdct::make_dct4_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f64(array.len()).unwrap();

            dct_forward
                .execute_into(&working_array, &mut transient)
                .expect(&format!("Failed to execute DCT-IV on size {i}"));
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .expect(&format!("Failed to execute DCT-IV on size {i}"));

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.00001, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst1_split_radix() {
        for i in 3usize..6 {
            let mut array = vec![0.; 2usize.pow(i as u32) + 1];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct1_f64(array.len()).unwrap();
            let naive_ref = naive_dct1(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct1_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct1_f64(array.len()).unwrap();
            let naive_ref = naive_dct1(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst1_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst1_f64(array.len()).unwrap();
            let naive_ref = naive_dst1(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct2_roundtrip_into() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f32;
            }
            let mut transient = vec![0f32; i];
            let mut working_array = vec![0f32; i];

            let dct_forward = Pxdct::make_dct2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f32(array.len()).unwrap();

            dct_forward.execute_into(&array, &mut transient).unwrap();
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct2_roundtrip_into_f64() {
        for i in 1..250 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut transient = vec![0f64; i];
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct2_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct3_f64(array.len()).unwrap();

            dct_forward.execute_into(&array, &mut transient).unwrap();
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst2_roundtrip_into() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f32;
            }
            let mut transient = vec![0f32; i];
            let mut working_array = vec![0f32; i];

            let dct_forward = Pxdct::make_dst2_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dst3_f32(array.len()).unwrap();

            dct_forward.execute_into(&array, &mut transient).unwrap();
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct4_roundtrip_into() {
        for i in 1..250 {
            let mut array = vec![0f32; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f32;
            }
            let mut transient = vec![0f32; i];
            let mut working_array = vec![0f32; i];

            let dct_forward = Pxdct::make_dct4_f32(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f32(array.len()).unwrap();

            dct_forward.execute_into(&array, &mut transient).unwrap();
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f32) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.01, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct4_roundtrip_into_f64() {
        for i in 1..300 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut transient = vec![0f64; i];
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct4_f64(array.len()).unwrap();
            let dct_inverse = Pxdct::make_dct4_f64(array.len()).unwrap();

            dct_forward.execute_into(&array, &mut transient).unwrap();
            dct_inverse
                .execute_into(&transient, &mut working_array)
                .unwrap();

            for k in working_array.iter_mut() {
                *k = *k / (i as f64) * 2.;
            }

            working_array.iter().zip(array.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 0.00001, "Difference to control values exceeded 0.00001 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct1_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct1_f64(array.len()).unwrap();
            let naive_ref = naive_dct1(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst1_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst1_f64(array.len()).unwrap();
            let naive_ref = naive_dst1(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst7_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst7_f64(array.len()).unwrap();
            let naive_ref = naive_dst7(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst7_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst7_f64(array.len()).unwrap();
            let naive_ref = naive_dst7(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct7_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct7_f64(array.len()).unwrap();
            let naive_ref = naive_dct7(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct7_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct7_f64(array.len()).unwrap();
            let naive_ref = naive_dct7(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dst4(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let angle = std::f64::consts::PI * (2 * i + 1) as f64 * (2 * k + 1) as f64
                            / (4 * n) as f64;
                        x * angle.sin()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn dst4_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst4_f64(array.len()).unwrap();
            let naive_ref = naive_dst4(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst4_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst4_f64(array.len()).unwrap();
            let naive_ref = naive_dst4(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst6_all() {
        for i in 2usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst6_f64(array.len()).unwrap();
            let naive_ref = naive_dst6(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst6_all_into() {
        for i in 2usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst6_f64(array.len()).unwrap();
            let naive_ref = naive_dst6(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dct6(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let cos_inner =
                    (output_index as f64) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64 - 0.5);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }

        result
    }

    #[test]
    fn dct6_all() {
        for i in 1usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct6_f64(array.len()).unwrap();
            let naive_ref = naive_dct6(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct6_all_into() {
        for i in 1usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct6_f64(array.len()).unwrap();
            let naive_ref = naive_dct6(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dct5(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner = (output_index as f64) * (input_index as f64) * std::f64::consts::PI
                    / (input.len() as f64 - 0.5);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }

        result
    }

    #[test]
    fn dct5_all() {
        // workaround stdarch bug
        for i in 30usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct5_f64(array.len()).unwrap();
            let naive_ref = naive_dct5(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct5_all_into() {
        for i in 1usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct5_f64(array.len()).unwrap();
            let naive_ref = naive_dct5(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dst5(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f64 + 1.0) * (input_index as f64 + 1.0) * std::f64::consts::PI
                        / (input.len() as f64 + 0.5);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }
        result
    }

    #[test]
    fn dst5_all() {
        for i in 1usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst5_f64(array.len()).unwrap();
            let naive_ref = naive_dst5(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst5_all_into() {
        for i in 1usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst5_f64(array.len()).unwrap();
            let naive_ref = naive_dst5(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dst8(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let sin_inner =
                    (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64 - 0.5);
                let twiddle = sin_inner.sin();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }

        result
    }

    #[test]
    fn dst8_all() {
        for i in 1usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dst8_f64(array.len()).unwrap();
            let naive_ref = naive_dst8(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dst8_all_into() {
        for i in 1usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dst8_f64(array.len()).unwrap();
            let naive_ref = naive_dst8(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    pub(crate) fn naive_dct8(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * std::f64::consts::PI
                        / (input.len() as f64 + 0.5);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle;
            }
            result.push(entry);
        }

        result
    }

    #[test]
    fn dct8_all() {
        for i in 1usize..150 {
            let mut array = vec![0.; i];
            for i in 1..i + 1 {
                array[i - 1] = i as f64;
            }
            let mut working_array = array.clone();
            let dct_forward = Pxdct::make_dct8_f64(array.len()).unwrap();
            let naive_ref = naive_dct8(&array);

            dct_forward.execute(&mut working_array).unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 0.01 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    #[test]
    fn dct8_all_into() {
        for i in 1usize..150 {
            let mut array = vec![0f64; i];
            for j in 1..i + 1 {
                array[j - 1] = j as f64;
            }
            let mut working_array = vec![0f64; i];

            let dct_forward = Pxdct::make_dct8_f64(array.len()).unwrap();
            let naive_ref = naive_dct8(&array);

            dct_forward
                .execute_into(&array, &mut working_array)
                .unwrap();

            working_array.iter().zip(naive_ref.iter()).enumerate().for_each(|(k, (&x, &c))| {
                assert!((x - c).abs() < 1e-7, "Difference to control values exceeded 1e-7 when it shouldn't, value {x}, control {c} at {k} for size {i}");
            });
        }
    }

    // -----------------------------------------------------------------------
    // f32 naive-reference helpers (mirrors the f64 versions above)
    // -----------------------------------------------------------------------

    pub(crate) fn naive_dst2_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f32 + 1.0) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32);
                entry += input[input_index] * sin_inner.sin();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst3_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let sin_inner =
                    (output_index as f32 + 0.5) * (input_index as f32 + 1.0) * std::f32::consts::PI
                        / (input.len() as f32);
                entry += input[input_index] * sin_inner.sin() * multiplier;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst4_f32(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let angle = std::f32::consts::PI * (2 * i + 1) as f32 * (2 * k + 1) as f32
                            / (4 * n) as f32;
                        x * angle.sin()
                    })
                    .sum()
            })
            .collect()
    }

    pub(crate) fn naive_dct5_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner = (output_index as f32) * (input_index as f32) * std::f32::consts::PI
                    / (input.len() as f32 - 0.5);
                entry += input[input_index] * cos_inner.cos() * multiplier;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst5_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f32 + 1.0) * (input_index as f32 + 1.0) * std::f32::consts::PI
                        / (input.len() as f32 + 0.5);
                entry += input[input_index] * sin_inner.sin();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dct6_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let cos_inner =
                    (output_index as f32) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32 - 0.5);
                entry += input[input_index] * cos_inner.cos() * multiplier;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst6_f32(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut result = Vec::with_capacity(n);
        for k in 0..n {
            let mut entry = 0.0_f32;
            for i in 0..n {
                let sin_inner = (k as f32 + 1.0) * (2.0 * i as f32 + 1.0) * std::f32::consts::PI
                    / (2.0 * n as f32 + 1.0);
                entry += input[i] * sin_inner.sin();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dct7_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
                let cos_inner =
                    (output_index as f32 + 0.5) * (input_index as f32) * std::f32::consts::PI
                        / (input.len() as f32 - 0.5);
                entry += input[input_index] * cos_inner.cos() * multiplier;
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst7_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let sin_inner =
                    (output_index as f32 + 0.5) * (input_index as f32 + 1.0) * std::f32::consts::PI
                        / (input.len() as f32 + 0.5);
                entry += input[input_index] * sin_inner.sin();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dct8_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let cos_inner =
                    (output_index as f32 + 0.5) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32 + 0.5);
                entry += input[input_index] * cos_inner.cos();
            }
            result.push(entry);
        }
        result
    }

    pub(crate) fn naive_dst8_f32(input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for output_index in 0..input.len() {
            let mut entry = 0.0_f32;
            for input_index in 0..input.len() {
                let multiplier = if input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let sin_inner =
                    (output_index as f32 + 0.5) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (input.len() as f32 - 0.5);
                entry += input[input_index] * sin_inner.sin() * multiplier;
            }
            result.push(entry);
        }
        result
    }

    // -----------------------------------------------------------------------
    // f32 in-place tests
    // -----------------------------------------------------------------------

    #[test]
    fn dst2_all_f32() {
        for i in 2usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst2_f32(&array);
            Pxdct::make_dst2_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst2_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst2_all_into_f32() {
        for i in 2usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst2_f32(&array);
            Pxdct::make_dst2_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst2_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst3_all_f32() {
        for i in 2usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst3_f32(&array);
            Pxdct::make_dst3_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst3_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst3_all_into_f32() {
        for i in 2usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst3_f32(&array);
            Pxdct::make_dst3_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst3_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst4_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst4_f32(&array);
            Pxdct::make_dst4_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst4_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst4_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst4_f32(&array);
            Pxdct::make_dst4_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst4_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct5_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dct5_f32(&array);
            Pxdct::make_dct5_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct5_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct5_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dct5_f32(&array);
            Pxdct::make_dct5_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct5_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst5_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst5_f32(&array);
            Pxdct::make_dst5_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst5_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst5_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst5_f32(&array);
            Pxdct::make_dst5_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst5_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct6_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dct6_f32(&array);
            Pxdct::make_dct6_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct6_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct6_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dct6_f32(&array);
            Pxdct::make_dct6_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct6_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst6_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst6_f32(&array);
            Pxdct::make_dst6_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst6_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst6_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst6_f32(&array);
            Pxdct::make_dst6_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst6_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct7_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dct7_f32(&array);
            Pxdct::make_dct7_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct7_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct7_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dct7_f32(&array);
            Pxdct::make_dct7_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct7_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst7_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst7_f32(&array);
            Pxdct::make_dst7_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst7_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst7_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst7_f32(&array);
            Pxdct::make_dst7_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst7_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct8_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dct8_f32(&array);
            Pxdct::make_dct8_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct8_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dct8_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dct8_f32(&array);
            Pxdct::make_dct8_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dct8_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst8_all_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut working = array.clone();
            let naive_ref = naive_dst8_f32(&array);
            Pxdct::make_dst8_f32(i)
                .unwrap()
                .execute(&mut working)
                .unwrap();
            working
                .iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst8_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }

    #[test]
    fn dst8_all_into_f32() {
        for i in 35usize..150 {
            let mut array = vec![0.0_f32; i];
            for j in 1..=i {
                array[j - 1] = j as f32;
            }
            let mut out = vec![0.0_f32; i];
            let naive_ref = naive_dst8_f32(&array);
            Pxdct::make_dst8_f32(i)
                .unwrap()
                .execute_into(&array, &mut out)
                .unwrap();
            out.iter()
                .zip(naive_ref.iter())
                .enumerate()
                .for_each(|(k, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-1,
                        "dst8_into_f32 mismatch at {k} size {i}: {x} vs {c}"
                    );
                });
        }
    }
}
