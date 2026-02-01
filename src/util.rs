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
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, BitXor, Mul, MulAssign, Neg};
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};

#[inline]
pub(crate) fn mixed_radix3_twiddles<T: DctSample + FftTrigonometry + 'static, const N: usize>(
    len: usize,
) -> [Complex<T>; N]
where
    f64: AsPrimitive<T>,
{
    let mut inner_layer = [Complex::<T>::default(); N];
    for (i, layer) in inner_layer.chunks_exact_mut(2).enumerate() {
        layer[0] = mixed_radix_inner_twiddle(2f64.as_() * (i as f64).as_() + 1f64.as_(), len);
        layer[0].im *= T::SQRT_3;
        layer[1] = mixed_radix_inner_twiddle(
            2f64.as_() * (2f64.as_() * (i as f64).as_() + 1f64.as_()),
            len,
        );
        layer[1].im = -layer[1].im * T::SQRT_3;
    }
    inner_layer
}

#[inline]
pub(crate) fn mixed_radix_inner_twiddle<T: Float + FftTrigonometry + 'static>(
    angle: T,
    fft_len: usize,
) -> Complex<T>
where
    f64: AsPrimitive<T>,
{
    let angle = angle / ((2 * fft_len) as f64).as_();
    let (v_sin, v_cos) = angle.sincos_pi();
    Complex {
        re: v_cos,
        im: v_sin,
    }
}

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
    + DctConstants
    + MixedRadix7Sample
    + MixedRadix5Sample
    + Neg<Output = Self>
    + TransposeFactory
    + AddAssign
    + Display
    + MixedRadix9Sample
    + MixedRadix17Sample
    + MixedRadix13Sample
    + MixedRadix19Sample
    + MixedRadix29Sample
    + MixedRadix23Sample
    + MulAssign
    + Send
    + Sync
{
    fn mulsigni(self, other: isize) -> Self;
    fn mulsign(self, other: Self) -> Self;
}

trait PointerToSinglePrecisionSize {
    fn to_single_precision_size(self) -> Self;
}

#[cfg(target_pointer_width = "64")]
impl PointerToSinglePrecisionSize for isize {
    #[inline(always)]
    fn to_single_precision_size(self) -> Self {
        self >> 32
    }
}

#[cfg(target_pointer_width = "32")]
impl PointerToSinglePrecisionSize for isize {
    #[inline(always)]
    fn to_single_precision_size(self) -> Self {
        self
    }
}

impl DctSample for f32 {
    #[inline(always)]
    fn mulsigni(self, other: isize) -> Self {
        let s_prec_size = other.to_single_precision_size();
        f32::from_bits(self.to_bits().bitxor((s_prec_size & (1isize << 31)) as u32))
    }

    #[inline(always)]
    fn mulsign(self, other: Self) -> Self {
        f32::from_bits(self.to_bits().bitxor(other.to_bits() & (1u32 << 31)))
    }
}

impl DctSample for f64 {
    #[inline(always)]
    fn mulsigni(self, other: isize) -> Self {
        let s_prec_size = other.to_single_precision_size();
        f64::from_bits(
            self.to_bits()
                .bitxor(((s_prec_size as i64) & (1i64 << 63)) as u64),
        )
    }

    #[inline(always)]
    fn mulsign(self, other: Self) -> Self {
        f64::from_bits(self.to_bits().bitxor(other.to_bits() & (1u64 << 63)))
    }
}

pub(crate) trait DctConstants {
    const HALF: Self;
    const TWO: Self;
    const SQRT_2: Self;
    const SQRT_3: Self;
    const FRAC_1_SQRT_2: Self;
    const SQRT_3_OVER_2: Self;
}

impl DctConstants for f32 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.;
    const SQRT_2: Self = std::f32::consts::SQRT_2;
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // float_to_hex(float(R(3).sqrt()))
    const SQRT_3: Self = f32::from_bits(0x3fddb3d7);
    const FRAC_1_SQRT_2: Self = std::f32::consts::FRAC_1_SQRT_2;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // value = R(3).sqrt() / R(2)
    //
    // print(float_to_hex(value))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(value))
    const SQRT_3_OVER_2: Self = f32::from_bits(0x3f5db3d7);
}

impl DctConstants for f64 {
    const HALF: Self = 0.5;
    const TWO: Self = 2.;
    const SQRT_2: Self = std::f64::consts::SQRT_2;
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // double_to_hex(float(R(3).sqrt()))
    const SQRT_3: Self = f64::from_bits(0x3ffbb67ae8584caa);
    const FRAC_1_SQRT_2: Self = std::f64::consts::FRAC_1_SQRT_2;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // value = R(3).sqrt() / R(2)
    //
    // print(float_to_hex(value))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(value))
    const SQRT_3_OVER_2: Self = f64::from_bits(0x3febb67ae8584caa);
}

pub(crate) trait FftProvider<T> {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, PxdctError>;

    fn make_fft_r2c(n: usize) -> Result<Arc<dyn R2CFftExecutor<T> + Send + Sync>, PxdctError>;
}

impl FftProvider<f32> for f32 {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, PxdctError> {
        match direction {
            FftDirection::Forward => Zaft::make_forward_fft_f32(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
            FftDirection::Inverse => Zaft::make_inverse_fft_f32(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
        }
    }

    fn make_fft_r2c(n: usize) -> Result<Arc<dyn R2CFftExecutor<f32> + Send + Sync>, PxdctError> {
        Zaft::make_r2c_fft_f32(n).map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string()))
    }
}

impl FftProvider<f64> for f64 {
    fn make_fft(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f64> + Send + Sync>, PxdctError> {
        match direction {
            FftDirection::Forward => Zaft::make_forward_fft_f64(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
            FftDirection::Inverse => Zaft::make_inverse_fft_f64(n)
                .map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string())),
        }
    }

    fn make_fft_r2c(n: usize) -> Result<Arc<dyn R2CFftExecutor<f64> + Send + Sync>, PxdctError> {
        Zaft::make_r2c_fft_f64(n).map_err(|x| PxdctError::CantCreateUnderlyingFft(x.to_string()))
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

macro_rules! define_butterfly {
    ($bf_name: ident, $length: expr) => {
        impl<T: DctSample> PxdctExecutor<T> for $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
                if !data.len().is_multiple_of($length) {
                    return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
                }
                for chunk in data.chunks_exact_mut($length) {
                    self.exec((&mut chunk[..$length]).try_into().unwrap());
                }
                Ok(())
            }

            fn length(&self) -> usize {
                $length
            }
        }
    };
}

pub(crate) use define_butterfly;

macro_rules! create_dct2_3 {
    ($clazz: ident) => {
        impl<T: DctSample> $clazz<T>
            where
                f64: AsPrimitive<T>,
            {
                pub(crate) fn new(len: usize) -> Result<$clazz<T>, PxdctError> {
                    let fft = T::make_fft(len, FftDirection::Forward)?;
                    use crate::twiddles::compute_twiddle;
                    let mut twiddles = try_vec![Complex::<T>::default(); len];
                    for (i, twiddle) in twiddles.iter_mut().enumerate() {
                        *twiddle = compute_twiddle::<T>(i, len * 4);
                    }

                    Ok($clazz {
                        twiddles,
                        fft_executor: fft,
                        execution_length: len,
                        spectrum_mul: T::create_mul_spectrum_to_real(),
                    })
                }
            }
    };
}

macro_rules! create_dct2_3_real {
    ($clazz: ident) => {
        impl<T: DctSample> $clazz<T>
            where
                f64: AsPrimitive<T>,
            {
                pub(crate) fn new(len: usize) -> Result<$clazz<T>, PxdctError> {
                    let fft = T::make_fft_r2c(len)?;
                    use crate::twiddles::compute_twiddle;
                    let mut twiddles = try_vec![Complex::<T>::default(); len];
                    for (i, twiddle) in twiddles.iter_mut().enumerate() {
                        *twiddle = compute_twiddle::<T>(i, len * 4);
                    }

                    Ok($clazz {
                        twiddles,
                        fft_executor: fft,
                        execution_length: len,
                        spectrum_mul: T::create_mul_spectrum_to_real(),
                    })
                }
            }
    };
}

use crate::butterflies::MixedRadix9Sample;
use crate::dct2::prime_butterflies::{
    MixedRadix13Sample, MixedRadix17Sample, MixedRadix19Sample, MixedRadix23Sample,
    MixedRadix29Sample,
};
use crate::dct2::{MixedRadix5Sample, MixedRadix7Sample};
use crate::transpose::TransposeFactory;
pub(crate) use create_dct2_3;
pub(crate) use create_dct2_3_real;
