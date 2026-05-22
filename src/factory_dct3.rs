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
use crate::type3::{
    Dct3Butterfly2, Dct3Butterfly3, Dct3Butterfly4, Dct3Butterfly5, Dct3Butterfly6, Dct3Butterfly7,
    Dct3Butterfly8, Dct3Butterfly9, Dct3Butterfly10, Dct3Butterfly11, Dct3Butterfly12,
    Dct3Butterfly13, Dct3Butterfly14, Dct3Butterfly15, Dct3Butterfly16, Dct3Butterfly18,
    Dct3Butterfly20, Dct3Butterfly21, Dct3Butterfly24, Dct3Butterfly26, Dct3Butterfly28,
    Dct3Butterfly30, Dct3Butterfly32, Dct3Butterfly35, Dct3Butterfly36, Dct3Butterfly64, Dct3Fft,
    Dct3Identity,
};
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use std::sync::{Arc, OnceLock};

pub(crate) trait Dct3Factory {
    fn dct3_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct3_fft(length: usize) -> Returning<Self>;
    fn dct3_mixed_radix3(inner_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>)
    -> Returning<Self>;
    fn dct3_mixed_radix5(inner_dct5: Arc<dyn PxdctExecutor<Self> + Send + Sync>)
    -> Returning<Self>;
    fn dct3_mixed_radix7(inner_dct7: Arc<dyn PxdctExecutor<Self> + Send + Sync>)
    -> Returning<Self>;
    fn dct3_mixed_radix9(inner_dct9: Arc<dyn PxdctExecutor<Self> + Send + Sync>)
    -> Returning<Self>;
    fn dct3_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct3_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dct3Factory for f32 {
    fn dct3_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonSplitRadixDct3f;
            Ok(Arc::new(NeonSplitRadixDct3f::new(
                length,
                half_dct,
                quarter_dct,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxSplitRadixDct3f;
            return Ok(Arc::new(AvxSplitRadixDct3f::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::SplitRadixDct3;
            Ok(Arc::new(SplitRadixDct3::new(
                length,
                half_dct,
                quarter_dct,
            )?))
        }
    }

    fn dct3_fft(length: usize) -> Returning<Self> {
        Ok(Arc::new(Dct3Fft::new(length)?))
    }

    fn dct3_mixed_radix3(
        inner_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix3f;
            Ok(Arc::new(NeonDct3MixedRadix3f::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix3f;
            return Ok(Arc::new(AvxDct3MixedRadix3f::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix3;
            Ok(Arc::new(Dct3MixedRadix3::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?))
        }
    }

    fn dct3_mixed_radix5(
        inner_dct5: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix5f;
            Ok(Arc::new(NeonDct3MixedRadix5f::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix5f;
            return Ok(Arc::new(AvxDct3MixedRadix5f::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix5;
            Ok(Arc::new(Dct3MixedRadix5::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?))
        }
    }

    fn dct3_mixed_radix7(
        inner_dct7: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix7f;
            Ok(Arc::new(NeonDct3MixedRadix7f::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix7f;
            return Ok(Arc::new(AvxDct3MixedRadix7f::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix7;
            Ok(Arc::new(Dct3MixedRadix7::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?))
        }
    }

    fn dct3_mixed_radix9(
        inner_dct9: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix9f;
            Ok(Arc::new(NeonDct3MixedRadix9f::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix9f;
            return Ok(Arc::new(AvxDct3MixedRadix9f::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix9;
            Ok(Arc::new(Dct3MixedRadix9::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?))
        }
    }

    fn dct3_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        Arc::new(Dct3Identity::default())
    }

    fn dct3_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly2::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly3::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly4::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly5::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly6::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly7::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly8::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly9::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly10::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly11::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly12::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly13::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly14::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly15::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly16;
                return Arc::new(AvxDct3Butterfly16::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly16::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly18::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly20::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly21::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly24::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly26::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly28::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly30::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly32;
                return Arc::new(AvxDct3Butterfly32::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly32::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly35::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly36::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly64;
                return Arc::new(AvxDct3Butterfly64::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly64::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }
}

impl Dct3Factory for f64 {
    fn dct3_split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonSplitRadixDct3d;
            Ok(Arc::new(NeonSplitRadixDct3d::new(
                length,
                half_dct,
                quarter_dct,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxSplitRadixDct3d;
            return Ok(Arc::new(AvxSplitRadixDct3d::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::SplitRadixDct3;
            Ok(Arc::new(SplitRadixDct3::new(
                length,
                half_dct,
                quarter_dct,
            )?))
        }
    }

    fn dct3_fft(length: usize) -> Returning<Self> {
        Ok(Arc::new(Dct3Fft::new(length)?))
    }

    fn dct3_mixed_radix3(
        inner_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix3d;
            Ok(Arc::new(NeonDct3MixedRadix3d::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix3d;
            return Ok(Arc::new(AvxDct3MixedRadix3d::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix3;
            Ok(Arc::new(Dct3MixedRadix3::new(
                inner_dct3.length() * 3,
                inner_dct3,
            )?))
        }
    }

    fn dct3_mixed_radix5(
        inner_dct5: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix5d;
            Ok(Arc::new(NeonDct3MixedRadix5d::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix5d;
            return Ok(Arc::new(AvxDct3MixedRadix5d::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix5;
            Ok(Arc::new(Dct3MixedRadix5::new(
                inner_dct5.length() * 5,
                inner_dct5,
            )?))
        }
    }

    fn dct3_mixed_radix7(
        inner_dct7: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix7d;
            Ok(Arc::new(NeonDct3MixedRadix7d::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix7d;
            return Ok(Arc::new(AvxDct3MixedRadix7d::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix7;
            Ok(Arc::new(Dct3MixedRadix7::new(
                inner_dct7.length() * 7,
                inner_dct7,
            )?))
        }
    }

    fn dct3_mixed_radix9(
        inner_dct9: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct3MixedRadix9d;
            Ok(Arc::new(NeonDct3MixedRadix9d::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct3MixedRadix9d;
            return Ok(Arc::new(AvxDct3MixedRadix9d::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type3::Dct3MixedRadix9;
            Ok(Arc::new(Dct3MixedRadix9::new(
                inner_dct9.length() * 9,
                inner_dct9,
            )?))
        }
    }

    fn dct3_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        Arc::new(Dct3Identity::default())
    }

    fn dct3_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly2::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly3::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly4::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly5::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly6::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly7::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly8::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly9::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly10::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly11::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly12::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly13::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly14::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly15::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly16;
                return Arc::new(AvxDct3Butterfly16::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly16::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly18::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly20::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly21::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly24::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly26::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly28::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly30::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly32;
                return Arc::new(AvxDct3Butterfly32::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly32::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly35::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct3Butterfly36::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct3_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct3Butterfly64;
                return Arc::new(AvxDct3Butterfly64::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct3Butterfly64::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }
}
