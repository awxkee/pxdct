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
use crate::mla::{c_mul_fast, fmla};
use crate::neon::util::NeonStoreF;
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::DctConstants;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vfcmulq_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    unsafe {
        let temp1 = vtrn1q_f32(rhs, rhs);
        let temp2 = vtrn2q_f32(rhs, vnegq_f32(rhs));
        let temp3 = vmulq_f32(temp2, lhs);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, lhs)
    }
}

#[inline]
pub(crate) unsafe fn reverse_f32(v: float32x4_t) -> float32x4_t {
    unsafe {
        let rev64 = vrev64q_f32(v);
        vcombine_f32(vget_high_f32(rev64), vget_low_f32(rev64))
    }
}

pub(crate) struct DctSpectrumMulF32 {}

impl DctSpectrumMul<f32> for DctSpectrumMulF32 {
    fn mul_spectrum_to_real_rev(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        for ((fft, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut().rev()) {
            *out = f32::mul_add(fft.re, twiddle.re, -fft.im * twiddle.im);
        }
    }

    fn mul_spectrum_to_real(
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

        let v_conj = NeonStoreF::set_values(0.0, -0.0, 0.0, -0.0);

        let mut i = 1usize;

        while i + 4 < complex_length {
            unsafe {
                let c0 = NeonStoreF::load_complex(complex_input.get_unchecked(i..));
                let c1 = NeonStoreF::load_complex(complex_input.get_unchecked(i + 2..));
                let twiddle0 = NeonStoreF::load_complex(twiddles.get_unchecked(i..));
                let twiddle1 = NeonStoreF::load_complex(twiddles.get_unchecked(i + 2..));
                let twiddle0_rev =
                    NeonStoreF::load_complex(twiddles.get_unchecked(len - i - 1..)).swap_complex();
                let twiddle1_rev =
                    NeonStoreF::load_complex(twiddles.get_unchecked(len - i - 3..)).swap_complex();

                let real_forward = NeonStoreF::mul_complex_unpack_real(c0, c1, twiddle0, twiddle1);
                let real_backward = NeonStoreF::mul_complex_unpack_real(
                    c0.xor(v_conj),
                    c1.xor(v_conj),
                    twiddle0_rev,
                    twiddle1_rev,
                );
                real_forward.write(out.get_unchecked_mut(i..));
                real_backward
                    .reverse()
                    .write(out.get_unchecked_mut(len - i - 3..));
            }

            i += 4;
        }

        for i in i..complex_length {
            unsafe {
                let twiddle = *twiddles.get_unchecked(i);
                let twiddle_rev = *twiddles.get_unchecked(len - i);
                let fft_value = *complex_input.get_unchecked(i);
                *out.get_unchecked_mut(i) =
                    fmla(fft_value.re, twiddle.re, -fft_value.im * twiddle.im);
                let conj_fft = fft_value.conj();
                *out.get_unchecked_mut(len - i) =
                    fmla(conj_fft.re, twiddle_rev.re, -conj_fft.im * twiddle_rev.im);
            }
        }
    }

    fn mul_spectrum_and_half(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
        out[0] = Complex::from(a[0] * f32::HALF);

        let a = &a[1..];
        let b = &b[1..];
        let out = &mut out[1..];

        let mut i = 0usize;

        let len = a.len();
        unsafe {
            while i + 4 < a.len() {
                let cf = vld1q_f32(a.get_unchecked(i..).as_ptr());
                let cb = reverse_f32(vld1q_f32(a.get_unchecked(len - i - 4..).as_ptr()));
                let tw0 = vld1q_f32(b.get_unchecked(i..).as_ptr().cast());
                let tw1 = vld1q_f32(b.get_unchecked(i + 2..).as_ptr().cast());

                let uq = vzipq_f32(cf, cb);

                let p0 = vmulq_n_f32(vfcmulq_f32(uq.0, tw0), 0.5);
                let p1 = vmulq_n_f32(vfcmulq_f32(uq.1, tw1), 0.5);

                vst1q_f32(out.get_unchecked_mut(i..).as_mut_ptr().cast(), p0);
                vst1q_f32(out.get_unchecked_mut(i + 2..).as_mut_ptr().cast(), p1);

                i += 4;
            }

            while i < len {
                let c = Complex {
                    re: *a.get_unchecked(i),
                    im: *a.get_unchecked(len - i - 1),
                };
                *out.get_unchecked_mut(i) = c_mul_fast(c, *b.get_unchecked(i)) * f32::HALF;
                i += 1;
            }
        }
    }

    fn mul_spectrum_and_half_rev(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
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
            *entry = c_mul_fast(c, *twiddle) * f32::HALF;
        }
    }
}
