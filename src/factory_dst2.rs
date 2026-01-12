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
use crate::PxdctExecutor;
use crate::factory_dct2::Returning;
use std::sync::Arc;

pub(crate) trait Dst2Factory {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
}

impl Dst2Factory for f32 {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && length >= 16
        {
            use crate::avx::AvxSplitRadixDst2f;
            return Ok(Arc::new(AvxSplitRadixDst2f::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if length >= 16 {
                use crate::neon::NeonSplitRadixDst2f;
                return Ok(Arc::new(NeonSplitRadixDst2f::new(
                    length,
                    half_dct,
                    quarter_dct,
                )?));
            }
        }
        use crate::dct2::SplitRadixDst2;
        Ok(Arc::new(SplitRadixDst2::new(
            length,
            half_dct,
            quarter_dct,
        )?))
    }
}

impl Dst2Factory for f64 {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && length > 16
        {
            use crate::avx::AvxSplitRadixDst2d;
            return Ok(Arc::new(AvxSplitRadixDst2d::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if length >= 16 {
                use crate::neon::NeonSplitRadixDst2d;
                return Ok(
                    Arc::new(NeonSplitRadixDst2d::new(length, half_dct, quarter_dct)?)
                        as Arc<dyn PxdctExecutor<f64> + Send + Sync>,
                );
            }
        }
        use crate::dct2::SplitRadixDst2;
        Ok(Arc::new(SplitRadixDst2::new(
            length,
            half_dct,
            quarter_dct,
        )?))
    }
}
