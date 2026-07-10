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
use crate::neon::mul_f32::reverse_f32;
use crate::neon::util::NeonStoreF;
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::DctConstants;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "fcma")]
pub(crate) unsafe fn vfcmul_fcma_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    vcmla_rot90_f32(vcmla_f32(vdup_n_f32(0.), lhs, rhs), lhs, rhs)
}

#[inline]
#[target_feature(enable = "fcma")]
pub(crate) unsafe fn vfcmulq_fcma_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), lhs, rhs), lhs, rhs)
}

pub(crate) struct FcmaDctSpectrumMulF32 {}

impl FcmaDctSpectrumMulF32 {
    #[target_feature(enable = "fcma")]
    fn mul_spectrum_and_half_impl(&self, a: &[f32], b: &[Complex<f32>], out: &mut [Complex<f32>]) {
        out[0] = Complex::from(a[0] * f32::HALF);

        let a = &a[1..];
        let b = &b[1..];
        let out = &mut out[1..];

        let mut i = 0usize;

        let len = a.len();
        unsafe {
            while i + 4 <= a.len() {
                let cf = vld1q_f32(a.get_unchecked(i..).as_ptr());
                let cb = reverse_f32(vld1q_f32(a.get_unchecked(len - i - 4..).as_ptr()));
                let tw0 = vld1q_f32(b.get_unchecked(i..).as_ptr().cast());
                let tw1 = vld1q_f32(b.get_unchecked(i + 2..).as_ptr().cast());

                let uq = vzipq_f32(cf, cb);

                let p0 = vmulq_n_f32(vfcmulq_fcma_f32(uq.0, tw0), 0.5);
                let p1 = vmulq_n_f32(vfcmulq_fcma_f32(uq.1, tw1), 0.5);

                vst1q_f32(out.get_unchecked_mut(i..).as_mut_ptr().cast(), p0);
                vst1q_f32(out.get_unchecked_mut(i + 2..).as_mut_ptr().cast(), p1);

                i += 4;
            }

            while i < len {
                let mut c =
                    vld1_lane_f32::<0>(a.get_unchecked(i..).as_ptr().cast(), vdup_n_f32(0.));
                c = vld1_lane_f32::<1>(a.get_unchecked(len - i - 1..).as_ptr().cast(), c);
                let tw = vld1_f32(b.get_unchecked(i..).as_ptr().cast());
                c = vmul_n_f32(vfcmul_fcma_f32(c, tw), 0.5);
                vst1_f32(out.get_unchecked_mut(i..).as_mut_ptr().cast(), c);
                i += 1;
            }
        }
    }
}

impl DctSpectrumMul<f32> for FcmaDctSpectrumMulF32 {
    fn mul_spectrum_to_real_rev(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        let out_len = out.len();
        let mut i = 0usize;
        for (fft, twiddle) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0.iter()) {
            unsafe {
                let a = vld1q_f32(fft.as_ptr().cast());
                let b = vld1q_f32(twiddle.as_ptr().cast());

                let a1 = vld1q_f32(fft[2..].as_ptr().cast());
                let b1 = vld1q_f32(twiddle[2..].as_ptr().cast());

                let a2 = vld1q_f32(fft[4..].as_ptr().cast());
                let b2 = vld1q_f32(twiddle[4..].as_ptr().cast());

                let a3 = vld1q_f32(fft[6..].as_ptr().cast());
                let b3 = vld1q_f32(twiddle[6..].as_ptr().cast());

                let a_z0 = vuzpq_f32(a, a1);
                let b_z0 = vuzpq_f32(b, b1);
                let a_re0 = a_z0.0;
                let a_im0 = a_z0.1;
                let b_re0 = b_z0.0;
                let b_im0 = b_z0.1;

                let a_z1 = vuzpq_f32(a2, a3);
                let b_z1 = vuzpq_f32(b2, b3);
                let a_re1 = a_z1.0;
                let a_im1 = a_z1.1;
                let b_re1 = b_z1.0;
                let b_im1 = b_z1.1;

                let real0 = vmlsq_f32(vmulq_f32(a_re0, b_re0), a_im0, b_im0);
                let real1 = vmlsq_f32(vmulq_f32(a_re1, b_re1), a_im1, b_im1);

                vst1q_f32(
                    out.get_unchecked_mut(out_len - i - 4..).as_mut_ptr(),
                    reverse_f32(real0),
                );
                vst1q_f32(
                    out.get_unchecked_mut(out_len - i - 8..).as_mut_ptr(),
                    reverse_f32(real1),
                );
                i += 8;
            }
        }

        let a = a.as_chunks::<8>().1;
        let b = b.as_chunks::<8>().1;

        for (fft, twiddle) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
            unsafe {
                let a = vld1q_f32(fft.as_ptr().cast());
                let b = vld1q_f32(twiddle.as_ptr().cast());

                let a1 = vld1q_f32(fft.get_unchecked(2..).as_ptr().cast());
                let b1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());

                let a_z0 = vuzpq_f32(a, a1);
                let b_z0 = vuzpq_f32(b, b1);
                let a_re0 = a_z0.0;
                let a_im0 = a_z0.1;
                let b_re0 = b_z0.0;
                let b_im0 = b_z0.1;
                let real = vmlsq_f32(vmulq_f32(a_re0, b_re0), a_im0, b_im0);
                vst1q_f32(
                    out.get_unchecked_mut(out_len - i - 4..).as_mut_ptr(),
                    reverse_f32(real),
                );
                i += 4;
            }
        }

        let a = a.as_chunks::<4>().1;
        let b = b.as_chunks::<4>().1;

        for (fft, twiddle) in a.iter().zip(b.iter()) {
            unsafe {
                *out.get_unchecked_mut(out_len - i - 1) =
                    f32::mul_add(fft.re, twiddle.re, -fft.im * twiddle.im);
                i += 1;
            }
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

        while i + 4 <= complex_length {
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
        unsafe { self.mul_spectrum_and_half_impl(a, b, out) }
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
