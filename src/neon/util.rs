/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::aarch64::*;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct NeonStoreF {
    pub(crate) v: float32x4_t,
}

impl Default for NeonStoreF {
    #[inline(always)]
    fn default() -> Self {
        unsafe { NeonStoreF::raw(vdupq_n_f32(0.)) }
    }
}

impl NeonStoreF {
    // #[inline(always)]
    // pub(crate) fn load_split2(ptr: &[f32]) -> [Self; 2] {
    //     let uq = unsafe { vld2q_f32(ptr.as_ptr()) };
    //     [NeonStoreF::raw(uq.0), NeonStoreF::raw(uq.1)]
    // }

    #[inline(always)]
    pub(crate) fn set_values(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        NeonStoreF::load(&[v0, v1, v2, v3])
    }

    #[inline(always)]
    pub(crate) fn prefix_differences(self, sign: NeonStoreF) -> Self {
        unsafe {
            let x = self.xor(sign);

            // prefix sum
            let r = vextq_f32::<3>(vdupq_n_f32(0.0), x.v);
            let s1 = vaddq_f32(x.v, r); // shift1
            let q = vextq_f32::<2>(vdupq_n_f32(0.0), s1);
            let s2 = vaddq_f32(s1, q); // shift2

            // undo sign
            NeonStoreF::raw(s2).xor(sign)
        }
    }

    #[inline(always)]
    pub(crate) fn broadcast_last(self) -> NeonStoreF {
        NeonStoreF::raw(unsafe { vdupq_laneq_f32::<3>(self.v) })
    }

    #[inline(always)]
    pub(crate) fn last(self) -> f32 {
        unsafe { vgetq_lane_f32::<3>(self.v) }
    }

    #[inline(always)]
    pub(crate) fn load(ptr: &[f32]) -> Self {
        NeonStoreF::raw(unsafe { vld1q_f32(ptr.as_ptr()) })
    }

    #[inline(always)]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        [
            NeonStoreF::raw(unsafe { vzip1q_f32(self.v, other.v) }),
            NeonStoreF::raw(unsafe { vzip2q_f32(self.v, other.v) }),
        ]
    }

    #[inline(always)]
    pub(crate) fn load_complex(ptr: &[Complex<f32>]) -> Self {
        NeonStoreF::raw(unsafe { vld1q_f32(ptr.as_ptr().cast()) })
    }

    #[inline(always)]
    pub(crate) fn mul_complex_unpack_real(
        a0: NeonStoreF,
        a1: NeonStoreF,
        b0: NeonStoreF,
        b1: NeonStoreF,
    ) -> Self {
        unsafe {
            let a_z0 = vuzpq_f32(a0.v, a1.v);
            let b_z0 = vuzpq_f32(b0.v, b1.v);
            let a_re0 = a_z0.0;
            let a_im0 = a_z0.1;
            let b_re0 = b_z0.0;
            let b_im0 = b_z0.1;
            let real = vmlsq_f32(vmulq_f32(a_re0, b_re0), a_im0, b_im0);
            NeonStoreF::raw(real)
        }
    }

    #[inline(always)]
    pub(crate) fn load3(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = vld1_f32(ptr.as_ptr());
            let q1 = vld1_lane_f32::<0>(ptr.get_unchecked(2..).as_ptr(), vdup_n_f32(0.));
            NeonStoreF::raw(vcombine_f32(q0, q1))
        }
    }

    #[inline(always)]
    pub(crate) fn load2(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = vld1_f32(ptr.as_ptr());
            NeonStoreF::raw(vcombine_f32(q0, vdup_n_f32(0.)))
        }
    }

    #[inline(always)]
    pub(crate) fn load1(ptr: &[f32]) -> Self {
        unsafe { NeonStoreF::raw(vld1q_lane_f32::<0>(ptr.as_ptr(), vdupq_n_f32(0.))) }
    }

    #[inline(always)]
    pub(crate) fn write(self, ptr: &mut [f32]) {
        unsafe { vst1q_f32(ptr.as_mut_ptr(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write3(self, ptr: &mut [f32]) {
        unsafe {
            vst1_f32(ptr.as_mut_ptr(), vget_low_f32(self.v));
            vst1q_lane_f32::<2>(ptr.get_unchecked_mut(2..).as_mut_ptr(), self.v);
        }
    }

    #[inline(always)]
    pub(crate) fn write2(self, ptr: &mut [f32]) {
        unsafe {
            vst1_f32(ptr.as_mut_ptr(), vget_low_f32(self.v));
        }
    }

    #[inline(always)]
    pub(crate) fn write1(self, ptr: &mut [f32]) {
        unsafe {
            vst1q_lane_f32::<0>(ptr.as_mut_ptr(), self.v);
        }
    }

    #[inline(always)]
    pub(crate) fn xor(&self, a1: NeonStoreF) -> Self {
        unsafe {
            NeonStoreF::raw(vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(self.v),
                vreinterpretq_u32_f32(a1.v),
            )))
        }
    }

    #[inline(always)]
    pub(crate) fn raw(v: float32x4_t) -> Self {
        NeonStoreF { v }
    }
}

impl Add<NeonStoreF> for NeonStoreF {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: NeonStoreF) -> Self::Output {
        NeonStoreF::raw(unsafe { vaddq_f32(self.v, rhs.v) })
    }
}

impl Sub<NeonStoreF> for NeonStoreF {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: NeonStoreF) -> Self::Output {
        NeonStoreF::raw(unsafe { vsubq_f32(self.v, rhs.v) })
    }
}

impl Mul<NeonStoreF> for NeonStoreF {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: NeonStoreF) -> Self::Output {
        NeonStoreF::raw(unsafe { vmulq_f32(self.v, rhs.v) })
    }
}

impl Mul<NeonStoreF> for f32 {
    type Output = NeonStoreF;
    #[inline(always)]
    fn mul(self, rhs: NeonStoreF) -> Self::Output {
        NeonStoreF::raw(unsafe { vmulq_n_f32(rhs.v, self) })
    }
}

impl Mul<f32> for NeonStoreF {
    type Output = NeonStoreF;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        NeonStoreF::raw(unsafe { vmulq_n_f32(self.v, rhs) })
    }
}

impl AddAssign for NeonStoreF {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            *self = NeonStoreF::raw(vaddq_f32(self.v, rhs.v));
        }
    }
}

impl Neg for NeonStoreF {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        NeonStoreF::raw(unsafe { vnegq_f32(self.v) })
    }
}

impl MulAdd<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: NeonStoreF, b: Self) -> Self::Output {
        NeonStoreF::raw(unsafe { vfmaq_f32(b.v, a.v, self.v) })
    }
}

impl NeonStoreF {
    #[inline(always)]
    pub(crate) fn f32_mul_add(q: f32, a: NeonStoreF, b: Self) -> Self {
        NeonStoreF::raw(unsafe { vfmaq_n_f32(b.v, a.v, q) })
    }

    #[inline(always)]
    pub(crate) fn f32_mul_nadd(q: f32, a: NeonStoreF, b: Self) -> Self {
        NeonStoreF::raw(unsafe { vfmsq_n_f32(b.v, a.v, q) })
    }

    #[inline(always)]
    pub(crate) fn reverse(self) -> Self {
        unsafe {
            let r = vrev64q_f32(self.v);
            NeonStoreF::raw(vcombine_f32(vget_high_f32(r), vget_low_f32(r)))
        }
    }

    #[inline(always)]
    pub(crate) fn swap_complex(self) -> Self {
        unsafe { NeonStoreF::raw(vcombine_f32(vget_high_f32(self.v), vget_low_f32(self.v))) }
    }

    #[inline(always)]
    pub(crate) fn reverse3(self) -> Self {
        unsafe {
            let shuffle_table: [u8; 16] = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15];
            let shuffle = vld1q_u8(shuffle_table.as_ptr());

            let ci = vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(self.v), shuffle));
            NeonStoreF::raw(ci)
        }
    }

    #[inline(always)]
    pub(crate) fn reverse2(self) -> Self {
        unsafe { NeonStoreF::raw(vrev64q_f32(self.v)) }
    }
}
