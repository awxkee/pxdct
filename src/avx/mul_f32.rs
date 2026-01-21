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
use crate::avx::storef::AvxStoreF;
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::DctConstants;
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
#[target_feature(enable = "avx2")]
fn _mm_unzip_ps(a: __m128, b: __m128) -> (__m128, __m128) {
    let v2 = _mm_unpacklo_ps(a, b); // a0 a2 b0 b2
    let v3 = _mm_unpackhi_ps(a, b); // a1 a3 b1 b3

    let va = _mm_unpacklo_ps(v2, v3); // a0 a1 a2 a3
    let vb = _mm_unpackhi_ps(v2, v3); // b0 b1 ab b3
    (va, vb)
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_ps(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(a); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(a); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_shuffle_ps::<0b10110001>(b, b); // [im, re, im, re, ...]

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
    fn mul_spectrum_to_real_impl(
        &self,
        complex_input: &[Complex<f32>],
        twiddles: &[Complex<f32>],
        out: &mut [f32],
    ) {
        let len = out.len();
        let half = out.len() / 2;
        let complex_length = complex_input.len();

        assert!(twiddles.len() >= len);
        assert!(!complex_input.is_empty());
        assert!(complex_input.len() >= half);
        assert!(out.len() >= len);

        unsafe {
            *out.get_unchecked_mut(0) =
                twiddles.get_unchecked(0).re * complex_input.get_unchecked(0).re;
            if len.is_multiple_of(2) {
                *out.get_unchecked_mut(half) =
                    twiddles.get_unchecked(half).re * complex_input.get_unchecked(half).re;
            }
        }

        let v_conj = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        let mut i = 1usize;

        while i + 8 < complex_length {
            unsafe {
                let c0 = AvxStoreF::load_complex(complex_input.get_unchecked(i..));
                let c1 = AvxStoreF::load_complex(complex_input.get_unchecked(i + 4..));
                let twiddle0 = AvxStoreF::load_complex(twiddles.get_unchecked(i..));
                let twiddle1 = AvxStoreF::load_complex(twiddles.get_unchecked(i + 4..));
                let twiddle0_rev =
                    AvxStoreF::load_complex(twiddles.get_unchecked(len - i - 3..)).swap_complex();
                let twiddle1_rev =
                    AvxStoreF::load_complex(twiddles.get_unchecked(len - i - 7..)).swap_complex();

                let real_forward =
                    AvxStoreF::mul_by_complex_unpack_real(c0, c1, twiddle0, twiddle1);
                let real_backward = AvxStoreF::mul_by_complex_unpack_real(
                    c0.xor(v_conj),
                    c1.xor(v_conj),
                    twiddle0_rev,
                    twiddle1_rev,
                );
                real_forward.write(out.get_unchecked_mut(i..));
                real_backward
                    .reverse()
                    .write(out.get_unchecked_mut(len - i - 7..));
            }

            i += 8;
        }

        for i in i..complex_length {
            unsafe {
                let twiddle = *twiddles.get_unchecked(i);
                let twiddle_rev = *twiddles.get_unchecked(len - i);
                let fft_value = *complex_input.get_unchecked(i);
                *out.get_unchecked_mut(i) =
                    f32::mul_add(fft_value.re, twiddle.re, -fft_value.im * twiddle.im);
                let conj_fft = fft_value.conj();
                *out.get_unchecked_mut(len - i) =
                    f32::mul_add(conj_fft.re, twiddle_rev.re, -conj_fft.im * twiddle_rev.im);
            }
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
                let q = _mm_loadu_ps(a.get_unchecked(len - i - 4..).as_ptr());
                let cb = _mm_shuffle_ps::<{ shuffle(0, 1, 2, 3) }>(q, q);
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
