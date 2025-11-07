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
use crate::mla::c_mul_fast;
use crate::util::DctSample;
use num_complex::Complex;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

pub(crate) trait DctSpectrumMul<T> {
    fn mul_spectrum_to_real(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [T]);
    fn mul_spectrum_to_real_rev(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [T]);
    fn mul_spectrum_and_half(&self, a: &[T], b: &[Complex<T>], out: &mut [Complex<T>]);
    fn mul_spectrum_and_half_rev(&self, a: &[T], b: &[Complex<T>], out: &mut [Complex<T>]);
}

pub(crate) struct FftSpectrumMul<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> DctSpectrumMul<T> for FftSpectrumMul<T> {
    fn mul_spectrum_to_real(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [T]) {
        for ((fft_entry, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out = c_mul_fast(*fft_entry, *twiddle).re;
        }
    }

    fn mul_spectrum_to_real_rev(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [T]) {
        for ((fft_entry, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut().rev()) {
            *out = c_mul_fast(*fft_entry, *twiddle).re;
        }
    }

    fn mul_spectrum_and_half(&self, a: &[T], b: &[Complex<T>], out: &mut [Complex<T>]) {
        out[0] = Complex::from(a[0] * T::HALF);

        for (((entry, twiddle), c_forward), c_backward) in out
            .iter_mut()
            .zip(b.iter())
            .zip(a.iter())
            .skip(1)
            .zip(a.iter().rev())
        {
            let c = Complex {
                re: *c_forward,
                im: *c_backward,
            };
            *entry = c_mul_fast(c, *twiddle) * T::HALF;
        }
    }

    fn mul_spectrum_and_half_rev(&self, a: &[T], b: &[Complex<T>], out: &mut [Complex<T>]) {
        let len_m1 = a.len() - 1;
        out[0] = Complex::from(a[len_m1] * T::HALF);

        for (((entry, twiddle), c_forward), c_backward) in out
            .iter_mut()
            .skip(1)
            .zip(b.iter().skip(1))
            .zip(a.iter())
            .zip(a.iter().rev().skip(1))
        {
            let c = Complex {
                re: *c_backward,
                im: *c_forward,
            };
            *entry = c_mul_fast(c, *twiddle) * T::HALF;
        }
    }
}

pub(crate) trait FftSpectrumMulFactory<T> {
    fn create_mul_spectrum_to_real() -> Arc<dyn DctSpectrumMul<T> + Send + Sync>;
}

impl FftSpectrumMulFactory<f32> for f32 {
    fn create_mul_spectrum_to_real() -> Arc<dyn DctSpectrumMul<f32> + Send + Sync> {
        static IMPL: OnceLock<Arc<dyn DctSpectrumMul<f32> + Send + Sync>> = OnceLock::new();
        IMPL.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                #[cfg(feature = "fcma")]
                {
                    if std::arch::is_aarch64_feature_detected!("fcma") {
                        use crate::neon::FcmaDctSpectrumMulF32;
                        return Arc::new(FcmaDctSpectrumMulF32 {});
                    }
                }
                use crate::neon::DctSpectrumMulF32;
                Arc::new(DctSpectrumMulF32 {})
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                use crate::util::has_valid_avx;
                if has_valid_avx() {
                    use crate::avx::AvxDctSpectrumMulF32;
                    return Arc::new(AvxDctSpectrumMulF32::default());
                }
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            Arc::new(FftSpectrumMul {
                phantom_data: PhantomData,
            })
        })
        .clone()
    }
}

impl FftSpectrumMulFactory<f64> for f64 {
    fn create_mul_spectrum_to_real() -> Arc<dyn DctSpectrumMul<f64> + Send + Sync> {
        Arc::new(FftSpectrumMul {
            phantom_data: PhantomData,
        })
    }
}
