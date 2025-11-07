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
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::Half;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[derive(Default)]
pub(crate) struct AvxDctSpectrumMulF32 {}

#[inline(always)]
fn avx_mul_fastf(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
    let re = f32::mul_add(a.re, b.re, -a.im * b.im);
    let im = f32::mul_add(a.re, b.im, a.im * b.re);
    Complex::new(re, im)
}

#[inline]
#[target_feature(enable = "avx")]
fn _mm_unzip_ps(a: __m128, b: __m128) -> (__m128, __m128) {
    let t0 = _mm_permute_ps::<{ shuffle(3, 1, 2, 0) }>(a);
    let t1 = _mm_permute_ps::<{ shuffle(3, 1, 2, 0) }>(b);

    // Now combine even and odd lanes:
    let o0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(t0), _mm_castps_pd(t1)));
    let o1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(t0), _mm_castps_pd(t1)));
    (o0, o1)
}

#[inline]
#[target_feature(enable = "avx2")]
fn _mm256_unzip_ps(a: __m256, b: __m256) -> (__m256, __m256) {
    let t0 = _mm256_permute_ps::<{ shuffle(3, 1, 2, 0) }>(a);
    let t1 = _mm256_permute_ps::<{ shuffle(3, 1, 2, 0) }>(b);

    // Now combine even and odd lanes:
    let o0 = _mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1));
    let o1 = _mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1));
    let u0 = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(o0));
    let u1 = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(o1));
    (u0, u1)
}

#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_ps(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(a); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(a); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_permute_ps::<0b10110001>(b); // [im, re, im, re, ...]

    // re = ar*br - ai*bi
    // im = ar*bi + ai*br
    _mm256_fmaddsub_ps(ar, b, _mm256_mul_ps(ai, bswap))
}

impl AvxDctSpectrumMulF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_spectrum_to_real_reversed_impl(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
        out: &mut [f32],
    ) {
        for ((fft, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut().rev()) {
            *out = f32::mul_add(fft.re, twiddle.re, -fft.im * twiddle.im);
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_spectrum_to_real_impl(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        for ((fft, twiddle), out) in a
            .chunks_exact(8)
            .zip(b.chunks_exact(8))
            .zip(out.chunks_exact_mut(8))
        {
            unsafe {
                let a = _mm256_loadu_ps(fft.as_ptr().cast());
                let b = _mm256_loadu_ps(twiddle.as_ptr().cast());

                let a2 = _mm256_loadu_ps(fft.get_unchecked(4..).as_ptr().cast());
                let b2 = _mm256_loadu_ps(twiddle.get_unchecked(4..).as_ptr().cast());

                let a_z0 = _mm256_unzip_ps(a, a2);
                let b_z0 = _mm256_unzip_ps(b, b2);
                let a_re0 = a_z0.0;
                let a_im0 = a_z0.1;
                let b_re0 = b_z0.0;
                let b_im0 = b_z0.1;

                let real0 = _mm256_fnmadd_ps(a_im0, b_im0, _mm256_mul_ps(a_re0, b_re0));

                _mm256_storeu_ps(out.as_mut_ptr(), real0);
            }
        }

        let a = a.chunks_exact(8).remainder();
        let b = b.chunks_exact(8).remainder();
        let out = out.chunks_exact_mut(8).into_remainder();

        for ((fft, twiddle), out) in a
            .chunks_exact(4)
            .zip(b.chunks_exact(4))
            .zip(out.chunks_exact_mut(4))
        {
            unsafe {
                let a = _mm256_loadu_ps(fft.as_ptr().cast());
                let b = _mm256_loadu_ps(twiddle.as_ptr().cast());

                let ka = _mm256_castps256_ps128(a);
                let kah = _mm256_extractf128_ps::<1>(a);

                let kb = _mm256_castps256_ps128(b);
                let kbh = _mm256_extractf128_ps::<1>(b);

                let a_z0 = _mm_unzip_ps(ka, kah);
                let b_z0 = _mm_unzip_ps(kb, kbh);
                let a_re0 = a_z0.0;
                let a_im0 = a_z0.1;
                let b_re0 = b_z0.0;
                let b_im0 = b_z0.1;
                let real = _mm_fnmadd_ps(a_im0, b_im0, _mm_mul_ps(a_re0, b_re0));
                _mm_storeu_ps(out.as_mut_ptr(), real);
            }
        }

        let a = a.chunks_exact(4).remainder();
        let b = b.chunks_exact(4).remainder();
        let out = out.chunks_exact_mut(4).into_remainder();

        for ((fft, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out = f32::mul_add(fft.re, twiddle.re, -fft.im * twiddle.im);
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_spectrum_and_half_impl(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
        out[0] = Complex::from(a[0] * f32::HALF);

        let a = &a[1..];
        let b = &b[1..];
        let out = &mut out[1..];

        let mut i = 0usize;

        let len = a.len();
        unsafe {
            let q_h = _mm256_set1_ps(0.5);

            while i + 4 < a.len() {
                let cf = _mm_loadu_ps(a.get_unchecked(i..).as_ptr());
                let cb = _mm_permute_ps::<{ shuffle(0, 1, 2, 3) }>(_mm_loadu_ps(
                    a.get_unchecked(len - i - 4..).as_ptr(),
                ));
                let tw0 = _mm256_loadu_ps(b.get_unchecked(i..).as_ptr().cast());

                let uq = (_mm_unpacklo_ps(cf, cb), _mm_unpackhi_ps(cf, cb));

                let p0 = _mm256_mul_ps(_mm256_fcmul_ps(_mm256_setr_m128(uq.0, uq.1), tw0), q_h);

                _mm256_storeu_ps(out.get_unchecked_mut(i..).as_mut_ptr().cast(), p0);

                i += 4;
            }

            while i < len {
                let c = Complex {
                    re: *a.get_unchecked(i),
                    im: *a.get_unchecked(len - i - 1),
                };
                *out.get_unchecked_mut(i) = avx_mul_fastf(c, *b.get_unchecked(i)) * f32::HALF;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_spectrum_and_half_rev_impl(
        &self,
        a: &[f32],
        b: &[Complex<f32>],
        out: &mut [Complex<f32>],
    ) {
        let len_m1 = a.len() - 1;
        out[0] = Complex::from(a[len_m1] * f32::HALF);

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
            *entry = avx_mul_fastf(c, *twiddle) * f32::HALF;
        }
    }
}

impl DctSpectrumMul<f32> for AvxDctSpectrumMulF32 {
    fn mul_spectrum_to_real_rev(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        unsafe { self.mul_spectrum_to_real_reversed_impl(a, b, out) }
    }

    fn mul_spectrum_to_real(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        unsafe { self.mul_spectrum_to_real_impl(a, b, out) }
    }

    fn mul_spectrum_and_half(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
        unsafe { self.mul_spectrum_and_half_impl(a, b, out) }
    }

    fn mul_spectrum_and_half_rev(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
        unsafe { self.mul_spectrum_and_half_rev_impl(a, b, out) }
    }
}
