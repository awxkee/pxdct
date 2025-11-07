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
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::Half;
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
    fn mul_spectrum_to_real(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [f32]) {
        for ((fft, twiddle), out) in a
            .chunks_exact(8)
            .zip(b.chunks_exact(8))
            .zip(out.chunks_exact_mut(8))
        {
            unsafe {
                let a = vld1q_f32(fft.as_ptr().cast());
                let b = vld1q_f32(twiddle.as_ptr().cast());

                let a1 = vld1q_f32(fft.get_unchecked(2..).as_ptr().cast());
                let b1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());

                let a2 = vld1q_f32(fft.get_unchecked(4..).as_ptr().cast());
                let b2 = vld1q_f32(twiddle.get_unchecked(4..).as_ptr().cast());

                let a3 = vld1q_f32(fft.get_unchecked(6..).as_ptr().cast());
                let b3 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());

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

                vst1q_f32(out.as_mut_ptr(), real0);
                vst1q_f32(out.get_unchecked_mut(4..).as_mut_ptr(), real1);
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
                vst1q_f32(out.as_mut_ptr(), real);
            }
        }

        let a = a.chunks_exact(4).remainder();
        let b = b.chunks_exact(4).remainder();
        let out = out.chunks_exact_mut(4).into_remainder();

        for ((fft, twiddle), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out = f32::mul_add(fft.re, twiddle.re, -fft.im * twiddle.im);
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
}
