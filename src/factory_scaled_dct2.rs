/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use std::sync::{Arc, OnceLock};

pub(crate) trait ScaledDct2Factory {
    fn scaled_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn scaled_dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn scaled_dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl ScaledDct2Factory for f32 {
    fn scaled_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::ScaledNeonSplitRadixDct2f;
            Ok(Arc::new(ScaledNeonSplitRadixDct2f::new(
                length,
                half_dct,
                quarter_dct,
            )?) as Arc<dyn PxdctExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::ScaledSplitRadixDct2;
            Ok(
                Arc::new(ScaledSplitRadixDct2::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>,
            )
        }
    }

    fn scaled_dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::power2_butterflies::Dct2Butterfly2;
            Arc::new(Dct2Butterfly2::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly4;
            Arc::new(ScaledDct2Butterfly4::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly8;
            Arc::new(ScaledDct2Butterfly8::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly16;
            Arc::new(ScaledDct2Butterfly16::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::ScaledNeonDct2Butterfly32f;
                Arc::new(ScaledNeonDct2Butterfly32f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::ScaledDct2Butterfly32;
                Arc::new(ScaledDct2Butterfly32::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn scaled_dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::ScaledNeonDct2Butterfly64f;
                Arc::new(ScaledNeonDct2Butterfly64f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::ScaledDct2Butterfly64;
                Arc::new(ScaledDct2Butterfly64::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn scaled_dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::ScaledNeonDct2Butterfly128f;
                Arc::new(ScaledNeonDct2Butterfly128f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::ScaledDct2Butterfly128;
                Arc::new(ScaledDct2Butterfly128::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn scaled_dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::ScaledNeonDct2Butterfly256f;
                Arc::new(ScaledNeonDct2Butterfly256f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::ScaledDct2Butterfly256;
                Arc::new(ScaledDct2Butterfly256::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn scaled_dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::ScaledNeonDct2Butterfly512f;
                Arc::new(ScaledNeonDct2Butterfly512f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::ScaledDct2Butterfly512;
                Arc::new(ScaledDct2Butterfly512::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }
}

impl ScaledDct2Factory for f64 {
    fn scaled_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::ScaledNeonSplitRadixDct2d;
            Ok(Arc::new(ScaledNeonSplitRadixDct2d::new(
                length,
                half_dct,
                quarter_dct,
            )?) as Arc<dyn PxdctExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::ScaledSplitRadixDct2;
            Ok(
                Arc::new(ScaledSplitRadixDct2::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>,
            )
        }
    }

    fn scaled_dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::power2_butterflies::Dct2Butterfly2;
            Arc::new(Dct2Butterfly2::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly4;
            Arc::new(ScaledDct2Butterfly4::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly8;
            Arc::new(ScaledDct2Butterfly8::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly16;
            Arc::new(ScaledDct2Butterfly16::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly32;
            Arc::new(ScaledDct2Butterfly32::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly64;
            Arc::new(ScaledDct2Butterfly64::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly128;
            Arc::new(ScaledDct2Butterfly128::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly256;
            Arc::new(ScaledDct2Butterfly256::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn scaled_dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::ScaledDct2Butterfly512;
            Arc::new(ScaledDct2Butterfly512::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }
}
