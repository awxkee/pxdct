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
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use std::sync::Arc;

pub(crate) trait Dst2Factory {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dst2_fft(length: usize) -> Returning<Self>;
    fn dst2_mixed_radix3(dst2_third: Arc<dyn PxdctExecutor<Self> + Send + Sync>)
    -> Returning<Self>;
    fn dst2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dst2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dst2Factory for f32 {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() && length >= 16 {
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
        use crate::type2::SplitRadixDst2;
        Ok(Arc::new(SplitRadixDst2::new(
            length,
            half_dct,
            quarter_dct,
        )?))
    }

    fn dst2_fft(length: usize) -> Returning<Self> {
        use crate::dst2::Dst2Fft;
        Ok(Arc::new(Dst2Fft::new(length)?))
    }

    fn dst2_mixed_radix3(
        dst2_third: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        use crate::type2::Dst2Radix3;
        Ok(Arc::new(Dst2Radix3::new(dst2_third)?))
    }

    fn dst2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dst2Butterfly2;
        Arc::new(Dst2Butterfly2::default())
    }

    fn dst2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly3;
        Arc::new(Dst2Butterfly3::default())
    }

    fn dst2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDst2Butterfly4;
            return Arc::new(AvxDst2Butterfly4::default());
        }
        use crate::type2::power2_butterflies::Dst2Butterfly4;
        Arc::new(Dst2Butterfly4::default())
    }

    fn dst2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly5;
        Arc::new(Dst2Butterfly5::default())
    }

    fn dst2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly6;
        Arc::new(Dst2Butterfly6::default())
    }

    fn dst2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly7;
        Arc::new(Dst2Butterfly7::default())
    }

    fn dst2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly8;
        Arc::new(Dst2Butterfly8::default())
    }

    fn dst2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly9;
        Arc::new(Dst2Butterfly9::default())
    }

    fn dst2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly16;
        Arc::new(Dst2Butterfly16::default())
    }
}

impl Dst2Factory for f64 {
    fn dst2_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() && length > 16 {
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
        use crate::type2::SplitRadixDst2;
        Ok(Arc::new(SplitRadixDst2::new(
            length,
            half_dct,
            quarter_dct,
        )?))
    }

    fn dst2_fft(length: usize) -> Returning<Self> {
        use crate::dst2::Dst2Fft;
        Ok(Arc::new(Dst2Fft::new(length)?))
    }

    fn dst2_mixed_radix3(
        dst2_third: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        use crate::type2::Dst2Radix3;
        Ok(Arc::new(Dst2Radix3::new(dst2_third)?))
    }

    fn dst2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dst2Butterfly2;
        Arc::new(Dst2Butterfly2::default())
    }

    fn dst2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly3;
        Arc::new(Dst2Butterfly3::default())
    }

    fn dst2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDst2Butterfly4;
            return Arc::new(AvxDst2Butterfly4::default());
        }
        use crate::type2::power2_butterflies::Dst2Butterfly4;
        Arc::new(Dst2Butterfly4::default())
    }

    fn dst2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly5;
        Arc::new(Dst2Butterfly5::default())
    }

    fn dst2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly6;
        Arc::new(Dst2Butterfly6::default())
    }

    fn dst2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly7;
        Arc::new(Dst2Butterfly7::default())
    }

    fn dst2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly8;
        Arc::new(Dst2Butterfly8::default())
    }

    fn dst2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly9;
        Arc::new(Dst2Butterfly9::default())
    }

    fn dst2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::Dst2Butterfly16;
        Arc::new(Dst2Butterfly16::default())
    }
}
