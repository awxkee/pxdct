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
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpackhilo_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(a, b)
}

#[inline(always)]
pub(crate) fn fma<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + MulAdd<T, Output = T>>(
    a: T,
    b: T,
    c: T,
) -> T {
    MulAdd::mul_add(a, b, c)
}

macro_rules! define_avx_butterfly {
    ($bf_name: ident, $length: expr) => {
        impl<T: DctSample> $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_impl(&self, data: &mut [T]) -> Result<(), PxdctError> {
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

        impl<T: DctSample> PxdctExecutor<T> for $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
                unsafe { self.execute_impl(data) }
            }

            fn length(&self) -> usize {
                $length
            }
        }
    };
}

pub(crate) use define_avx_butterfly;
