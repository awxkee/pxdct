/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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
use num_traits::MulAdd;
use std::arch::aarch64::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct NeonStoreD {
    pub(crate) v: float64x2_t,
}

impl Default for NeonStoreD {
    #[inline(always)]
    fn default() -> Self {
        unsafe { NeonStoreD::raw(vdupq_n_f64(0.)) }
    }
}

impl NeonStoreD {
    #[inline(always)]
    pub(crate) fn set_values(v0: f64, v1: f64) -> Self {
        NeonStoreD {
            v: unsafe { vld1q_f64([v0, v1].as_ptr()) },
        }
    }

    #[inline(always)]
    pub(crate) fn dup(v0: f64) -> Self {
        NeonStoreD {
            v: unsafe { vdupq_n_f64(v0) },
        }
    }

    #[inline(always)]
    pub(crate) fn raw(v: float64x2_t) -> Self {
        NeonStoreD { v }
    }

    #[inline(always)]
    pub(crate) fn load(ptr: &[f64]) -> Self {
        NeonStoreD::raw(unsafe { vld1q_f64(ptr.as_ptr()) })
    }

    #[inline(always)]
    pub(crate) fn load_n<const N: usize>(ptr: &[f64]) -> Self {
        const {
            assert!(N <= 2, "N must be <= 2");
        }
        match N {
            2 => Self::load(ptr),
            _ => Self::load1(ptr),
        }
    }

    #[inline(always)]
    pub(crate) fn load1(ptr: &[f64]) -> Self {
        NeonStoreD::raw(unsafe { vld1q_lane_f64::<0>(ptr.as_ptr(), vdupq_n_f64(0.)) })
    }

    #[inline(always)]
    pub(crate) fn write(self, ptr: &mut [f64]) {
        unsafe { vst1q_f64(ptr.as_mut_ptr(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_n<const N: usize>(self, ptr: &mut [f64]) {
        const {
            assert!(N <= 2, "N must be <= 2");
        }
        match N {
            2 => self.write(ptr),
            _ => self.write1(ptr),
        }
    }

    #[inline(always)]
    pub(crate) fn write1(self, ptr: &mut [f64]) {
        unsafe { vst1q_lane_f64::<0>(ptr.as_mut_ptr(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn reverse(self) -> Self {
        unsafe { NeonStoreD::raw(vcombine_f64(vget_high_f64(self.v), vget_low_f64(self.v))) }
    }

    #[inline(always)]
    pub(crate) fn reverse_n<const N: usize>(self) -> Self {
        const {
            assert!(N <= 2, "N must be <= 2");
        }
        match N {
            2 => self.reverse(),
            _ => self,
        }
    }

    #[inline(always)]
    pub(crate) fn f64_mul_add(q: f64, a: NeonStoreD, b: Self) -> Self {
        NeonStoreD::raw(unsafe { vfmaq_n_f64(b.v, a.v, q) })
    }

    #[inline(always)]
    pub(crate) fn mul_f64_add(p0: NeonStoreD, p1: f64, p2: NeonStoreD) -> NeonStoreD {
        NeonStoreD::raw(unsafe { vfmaq_n_f64(p2.v, p0.v, p1) })
    }

    #[inline(always)]
    pub(crate) fn f64_mul_nadd(q: f64, a: NeonStoreD, b: NeonStoreD) -> NeonStoreD {
        NeonStoreD::raw(unsafe { vfmsq_n_f64(b.v, a.v, q) })
    }

    #[inline(always)]
    pub(crate) fn xor(&self, p0: NeonStoreD) -> NeonStoreD {
        NeonStoreD::raw(unsafe {
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(self.v),
                vreinterpretq_u64_f64(p0.v),
            ))
        })
    }

    #[inline(always)]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        [
            NeonStoreD::raw(unsafe { vzip1q_f64(self.v, other.v) }),
            NeonStoreD::raw(unsafe { vzip2q_f64(self.v, other.v) }),
        ]
    }
}

impl Add<NeonStoreD> for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: NeonStoreD) -> Self::Output {
        NeonStoreD::raw(unsafe { vaddq_f64(self.v, rhs.v) })
    }
}

impl AddAssign for NeonStoreD {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = NeonStoreD::raw(unsafe { vaddq_f64(self.v, rhs.v) })
    }
}

impl Sub<NeonStoreD> for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: NeonStoreD) -> Self::Output {
        NeonStoreD::raw(unsafe { vsubq_f64(self.v, rhs.v) })
    }
}

impl Neg for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        NeonStoreD::raw(unsafe { vnegq_f64(self.v) })
    }
}

impl Mul<NeonStoreD> for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: NeonStoreD) -> Self::Output {
        NeonStoreD::raw(unsafe { vmulq_f64(self.v, rhs.v) })
    }
}

impl Mul<f64> for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        NeonStoreD::raw(unsafe { vmulq_n_f64(self.v, rhs) })
    }
}

impl MulAssign<f64> for NeonStoreD {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        *self = NeonStoreD::raw(unsafe { vmulq_n_f64(self.v, rhs) })
    }
}

impl Mul<NeonStoreD> for f64 {
    type Output = NeonStoreD;
    #[inline(always)]
    fn mul(self, rhs: NeonStoreD) -> Self::Output {
        NeonStoreD::raw(unsafe { vmulq_n_f64(rhs.v, self) })
    }
}

impl MulAdd<NeonStoreD> for NeonStoreD {
    type Output = Self;
    #[inline(always)]
    fn mul_add(self, a: NeonStoreD, b: Self) -> Self::Output {
        Self::raw(unsafe { vfmaq_f64(b.v, self.v, a.v) })
    }
}
