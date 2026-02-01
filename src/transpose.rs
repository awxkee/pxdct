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
use std::sync::Arc;

pub(crate) trait Transposition<T> {
    fn transpose(&self, src: &[T], dst: &mut [T]);
}

pub(crate) trait TransposeFactory: Sized {
    fn make_transpose(width: usize, height: usize) -> Arc<dyn Transposition<Self> + Send + Sync>;
}

pub(crate) struct TransposeTiny {
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<T: Copy> Transposition<T> for TransposeTiny {
    fn transpose(&self, src: &[T], dst: &mut [T]) {
        for x in 0..self.width {
            for y in 0..self.height {
                let input_index = x + y * self.width;
                let output_index = y + x * self.height;

                unsafe {
                    *dst.get_unchecked_mut(output_index) = *src.get_unchecked(input_index);
                }
            }
        }
    }
}

impl TransposeFactory for f32 {
    fn make_transpose(width: usize, height: usize) -> Arc<dyn Transposition<Self> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if height.is_multiple_of(11) {
                use crate::neon::NeonTransposeNx11F32;
                return Arc::new(NeonTransposeNx11F32 { width, height });
            }
            if height.is_multiple_of(7) {
                use crate::neon::NeonTransposeNx7F32;
                return Arc::new(NeonTransposeNx7F32 { width, height });
            }
            if height.is_multiple_of(6) {
                use crate::neon::NeonTransposeNx6F32;
                return Arc::new(NeonTransposeNx6F32 { width, height });
            }
            if height.is_multiple_of(5) {
                use crate::neon::NeonTransposeNx5F32;
                return Arc::new(NeonTransposeNx5F32 { width, height });
            }
            use crate::neon::NeonTranspose4x4;
            Arc::new(NeonTranspose4x4 { width, height })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::util::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxTransposeFReal4x4;
                return Arc::new(AvxTransposeFReal4x4 { width, height });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(TransposeTiny { width, height })
        }
    }
}

impl TransposeFactory for f64 {
    fn make_transpose(width: usize, height: usize) -> Arc<dyn Transposition<Self> + Send + Sync> {
        Arc::new(TransposeTiny { width, height })
    }
}
