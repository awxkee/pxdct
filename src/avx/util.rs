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
                use crate::bidirectional::InPlaceStore;
                if !data.len().is_multiple_of($length) {
                    return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
                }
                for chunk in data.chunks_exact_mut($length) {
                    self.exec(&mut InPlaceStore::new(chunk));
                }
                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_into_impl(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(input, output, $length);
                use crate::bidirectional::BiStore;
                for (src, dst) in input
                    .chunks_exact($length)
                    .zip(output.chunks_exact_mut($length))
                {
                    self.exec(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl<T: DctSample> PxdctExecutor<T> for $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
                unsafe { self.execute_impl(data) }
            }

            fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
                unsafe { self.execute_impl(data) }
            }

            fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
                unsafe { self.execute_into_impl(input, output) }
            }

            fn execute_into_with_scratch(
                &self,
                input: &[T],
                output: &mut [T],
                _: &mut [T],
            ) -> Result<(), PxdctError> {
                unsafe { self.execute_into_impl(input, output) }
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

pub(crate) use define_avx_butterfly;

macro_rules! boring_avx_mixed_radix {
    ($f_name: ident, $f_type: ident) => {
        impl $f_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_with_scratch_impl(
                &self,
                data: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                if !data.len().is_multiple_of(self.execution_length) {
                    return Err(PxdctError::InvalidSizeMultiplier(
                        data.len(),
                        self.execution_length,
                    ));
                }

                use crate::util::validate_scratch;
                let full_scratch = validate_scratch!(scratch, self.scratch_size());

               use crate::bidirectional::InPlaceStore;
                for chunk in data.chunks_exact_mut(self.execution_length) {
                    self.execute_store(&mut InPlaceStore::new(chunk), full_scratch)?;
                }

                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_into_with_scratch_impl(
                &self,
                input: &[$f_type],
                output: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(input, output, self.execution_length);

                let full_scratch = validate_scratch!(scratch, self.scratch_size());

                use crate::bidirectional::BiStore;
                for (src, dst) in input.chunks_exact(self.execution_length).zip(output.chunks_exact_mut(self.execution_length)) {
                    self.execute_store(&mut BiStore::new(src, dst), full_scratch)?;
                }
                Ok(())
            }

        }

        impl PxdctExecutor<$f_type> for $f_name {
            fn execute(&self, data: &mut [$f_type]) -> Result<(), PxdctError> {
                let mut scratch = try_vec![$f_type::default(); self.scratch_size()];
                unsafe { self.execute_with_scratch_impl(data, &mut scratch) }
            }

            fn execute_with_scratch(
                &self,
                data: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                unsafe { self.execute_with_scratch_impl(data, scratch) }
            }

            fn execute_into(&self, input: &[$f_type], output: &mut [$f_type]) -> Result<(), PxdctError> {
                let mut scratch = try_vec![$f_type::default(); self.scratch_size()];
                unsafe { self.execute_into_with_scratch_impl(input, output, &mut scratch) }
            }

            fn execute_into_with_scratch(
                &self,
                input: &[$f_type],
                output: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                unsafe { self.execute_into_with_scratch_impl(input, output, scratch) }
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }

            #[inline]
            fn scratch_size(&self) -> usize {
                self.execution_length + self.inner_dct_scratch_size
            }
}
    };
}

pub(crate) use boring_avx_mixed_radix;

macro_rules! boring_avx_split_radix {
    ($f_name: ident, $f_type: ident) => {
        impl $f_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_with_scratch_impl(
                &self,
                data: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                if !data.len().is_multiple_of(self.execution_length) {
                    return Err(PxdctError::InvalidSizeMultiplier(
                        data.len(),
                        self.execution_length,
                    ));
                }

                use crate::util::validate_scratch;
                let full_scratch = validate_scratch!(scratch, self.scratch_size());

               use crate::bidirectional::InPlaceStore;
                for chunk in data.chunks_exact_mut(self.execution_length) {
                    self.execute_store(&mut InPlaceStore::new(chunk), full_scratch)?;
                }

                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_into_with_scratch_impl(
                &self,
                input: &[$f_type],
                output: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(input, output, self.execution_length);

                let full_scratch = validate_scratch!(scratch, self.scratch_size());

                use crate::bidirectional::BiStore;
                for (src, dst) in input.chunks_exact(self.execution_length).zip(output.chunks_exact_mut(self.execution_length)) {
                    self.execute_store(&mut BiStore::new(src, dst), full_scratch)?;
                }
                Ok(())
            }

        }

        impl PxdctExecutor<$f_type> for $f_name {
            fn execute(&self, data: &mut [$f_type]) -> Result<(), PxdctError> {
                let mut scratch = try_vec![$f_type::default(); self.scratch_size()];
                unsafe { self.execute_with_scratch_impl(data, &mut scratch) }
            }

            fn execute_with_scratch(
                &self,
                data: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                unsafe { self.execute_with_scratch_impl(data, scratch) }
            }

            fn execute_into(&self, input: &[$f_type], output: &mut [$f_type]) -> Result<(), PxdctError> {
                let mut scratch = try_vec![$f_type::default(); self.scratch_size()];
                unsafe { self.execute_into_with_scratch_impl(input, output, &mut scratch) }
            }

            fn execute_into_with_scratch(
                &self,
                input: &[$f_type],
                output: &mut [$f_type],
                scratch: &mut [$f_type],
            ) -> Result<(), PxdctError> {
                unsafe { self.execute_into_with_scratch_impl(input, output, scratch) }
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }

            #[inline]
            fn scratch_size(&self) -> usize {
                self.inner_scratch_size
            }
}
    };
}

pub(crate) use boring_avx_split_radix;
