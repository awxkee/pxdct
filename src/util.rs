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
    + Dct4MixedRadix9Sample
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
    const QUARTER: Self;
    const ONE_EIGHT: Self;
    const TWO: Self;
    const SQRT_2: Self;
    const SQRT_3: Self;
    const FRAC_1_SQRT_2: Self;
    const SQRT_3_OVER_2: Self;
    const SQRT_2_OVER_16: Self;
    const SQRT_2_OVER_64: Self;
    const SQRT_2_OVER_256: Self;
    const SQRT_2_OVER_512: Self;
}

impl DctConstants for f32 {
    const HALF: Self = 0.5;
    const QUARTER: Self = 0.25;
    const ONE_EIGHT: Self = 0.125;
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
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float((R(2)/R(16)).sqrt())))
    const SQRT_2_OVER_16: Self = f32::from_bits(0x3eb504f3);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float((R(2)/R(64)).sqrt())))
    const SQRT_2_OVER_64: Self = f32::from_bits(0x3e3504f3);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float((R(2)/R(256)).sqrt())))
    const SQRT_2_OVER_256: Self = f32::from_bits(0x3db504f3);
    const SQRT_2_OVER_512: Self = 0.0625;
}

impl DctConstants for f64 {
    const HALF: Self = 0.5;
    const QUARTER: Self = 0.25;
    const ONE_EIGHT: Self = 0.125;
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
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float((R(2)/R(16)).sqrt())))
    const SQRT_2_OVER_16: Self = f64::from_bits(0x3fd6a09e667f3bcd);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float((R(2)/R(64)).sqrt())))
    const SQRT_2_OVER_64: Self = f64::from_bits(0x3fc6a09e667f3bcd);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float((R(2)/R(256)).sqrt())))
    const SQRT_2_OVER_256: Self = f64::from_bits(0x3fb6a09e667f3bcd);
    const SQRT_2_OVER_512: Self = 0.0625;
}

macro_rules! validate_oof_sizes {
    ($src: expr, $dst: expr, $length: expr) => {{
        if !$src.len().is_multiple_of($length) {
            return Err(PxdctError::InvalidSizeMultiplier($src.len(), $length));
        }
        if !$dst.len().is_multiple_of($length) {
            return Err(PxdctError::InvalidSizeMultiplier($dst.len(), $length));
        }
        if $dst.len() != $src.len() {
            return Err(PxdctError::OutOfPlaceSizeDoesntMatch(
                $src.len(),
                $dst.len(),
            ));
        }
    }};
}

pub(crate) use validate_oof_sizes;

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

macro_rules! validate_scratch {
    ($scratch: expr, $size: expr) => {{
        if $scratch.len() < $size {
            return Err(crate::PxdctError::InvalidScratchSize($size, $scratch.len()));
        }
        let (left, _) = $scratch.split_at_mut($size);
        left
    }};
}

use crate::spectrum_mul::FftSpectrumMulFactory;
pub(crate) use try_vec;
pub(crate) use validate_scratch;

#[inline]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(crate) fn has_valid_avx() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

macro_rules! define_in_place_butterfly {
    ($bf_name: ident, $length: expr) => {
        impl<T: DctSample> PxdctExecutor<T> for $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
                if !data.len().is_multiple_of($length) {
                    return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
                }
                use crate::bidirectional::InPlaceStore;
                for chunk in data.as_chunks_mut::<$length>().0.iter_mut() {
                    self.exec(&mut InPlaceStore::new(chunk));
                }
                Ok(())
            }

            fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
                if !data.len().is_multiple_of($length) {
                    return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
                }
                use crate::bidirectional::InPlaceStore;
                for chunk in data.as_chunks_mut::<$length>().0.iter_mut() {
                    self.exec(&mut InPlaceStore::new(chunk));
                }
                Ok(())
            }

            fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
                self.execute_into_with_scratch(input, output, &mut [])
            }

            fn execute_into_with_scratch(
                &self,
                input: &[T],
                output: &mut [T],
                _: &mut [T],
            ) -> Result<(), PxdctError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(input, output, $length);
                use crate::bidirectional::BiStore;
                for (src, dst) in input
                    .as_chunks::<$length>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<$length>().0.iter_mut())
                {
                    self.exec(&mut BiStore::new(src, dst));
                }
                Ok(())
            }

            fn length(&self) -> usize {
                $length
            }

            fn scratch_size(&self) -> usize {
                0
            }
        }
    };
}

pub(crate) use define_in_place_butterfly;

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

                    let fft_scratch_size = fft.scratch_length();

                    Ok($clazz {
                        twiddles,
                        fft_executor: fft,
                        execution_length: len,
                        spectrum_mul: T::create_mul_spectrum_to_real(),
                        fft_scratch_size,
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

                    let fft_scratch_size = fft.complex_scratch_length();

                    Ok($clazz {
                        twiddles,
                        fft_executor: fft,
                        execution_length: len,
                        spectrum_mul: T::create_mul_spectrum_to_real(),
                        fft_scratch_size,
                    })
                }
            }
    };
}

use crate::butterflies::MixedRadix9Sample;
use crate::transpose::TransposeFactory;
use crate::type2::prime_butterflies::{
    MixedRadix13Sample, MixedRadix17Sample, MixedRadix19Sample, MixedRadix23Sample,
    MixedRadix29Sample,
};
use crate::type2::{MixedRadix5Sample, MixedRadix7Sample};
use crate::type4::Dct4MixedRadix9Sample;
pub(crate) use create_dct2_3;
pub(crate) use create_dct2_3_real;

pub(crate) fn force_cast_real_scratch_to_complex<T: Copy + Sized>(
    source: &mut [T],
    new_complex_scratch_size: usize,
) -> &mut [Complex<T>] {
    // check if algorithm pre-request are valid
    assert_eq!(align_of::<T>(), align_of::<Complex<T>>());
    assert_eq!(size_of::<Complex<T>>(), 2 * size_of::<T>());
    // size should be enough, no validate alignment
    assert!(source.len() >= new_complex_scratch_size * 2);
    // if align of source.ptr() is not valid to cast it to Complex<T> then we'll move
    // one element forward and this must match the required alignment, are stated above
    unsafe {
        if source
            .as_ptr()
            .addr()
            .is_multiple_of(align_of::<Complex<T>>())
        {
            std::slice::from_raw_parts_mut(
                source.as_mut_ptr() as *mut Complex<T>,
                new_complex_scratch_size,
            )
        } else {
            panic!("Alignment check failed, that must not happen");
        }
    }
}
