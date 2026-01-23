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
use crate::dct4::{
    Dct4Butterfly2, Dct4Butterfly3, Dct4Butterfly4, Dct4Butterfly6, Dct4Butterfly8,
    Dct4Butterfly10, Dct4Butterfly12, Dct4Butterfly14, Dct4Butterfly16, Dct4Butterfly18,
    Dct4Butterfly20, Dct4Butterfly22, Dct4Butterfly24, Dct4Butterfly26, Dct4Butterfly28,
    Dct4Butterfly30, Dct4Butterfly32, Dct4Fft, Dct4Identity, Dct4Radix2,
};
use crate::factory_dct2::Returning;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use std::sync::{Arc, OnceLock};
use zaft::FftExecutor;

pub(crate) trait Dct4Factory {
    fn dct4_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix3(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix5(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix7(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix9(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix11(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix13(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix17(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_mixed_radix19(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct4_fft(fft: Arc<dyn FftExecutor<Self> + Send + Sync>) -> Returning<Self>;
    fn dct4_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dct4Factory for f32 {
    fn dct4_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4Radix2f;
            return Ok(Arc::new(AvxDct4Radix2f::new(len, half_dct2)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4Radix2f;
            Ok(Arc::new(NeonDct4Radix2f::new(len, half_dct2)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Ok(Arc::new(Dct4Radix2::new(len, half_dct2)?))
        }
    }

    fn dct4_mixed_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix2f;
            return Ok(Arc::new(AvxDct4MixedRadix2f::new(len, half_dct2)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix2f;
            Ok(Arc::new(NeonDct4MixedRadix2f::new(len, half_dct2)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix2;
            Ok(Arc::new(Dct4MixedRadix2::new(len, half_dct2)?))
        }
    }

    fn dct4_mixed_radix3(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix3f;
            return Ok(Arc::new(AvxDct4MixedRadix3f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix3f;
            Ok(Arc::new(NeonDct4MixedRadix3f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix3;
            Ok(Arc::new(Dct4MixedRadix3::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix5(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix5f;
            return Ok(Arc::new(AvxDct4MixedRadix5f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix5f;
            Ok(Arc::new(NeonDct4MixedRadix5f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix5;
            Ok(Arc::new(Dct4MixedRadix5::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix7(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix7f;
            return Ok(Arc::new(AvxDct4MixedRadix7f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix7f;
            Ok(Arc::new(NeonDct4MixedRadix7f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix7;
            Ok(Arc::new(Dct4MixedRadix7::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix9(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix9f;
            Ok(Arc::new(NeonDct4MixedRadix9f::new(len, half_dct4)?))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix9f;
            return Ok(Arc::new(AvxDct4MixedRadix9f::new(len, half_dct4)?));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix9;
            Ok(Arc::new(Dct4MixedRadix9::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix11(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix11f;
            return Ok(Arc::new(AvxDct4MixedRadix11f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix11f;
            Ok(Arc::new(NeonDct4MixedRadix11f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix11;
            Ok(Arc::new(Dct4MixedRadix11::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix13(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix13f;
            return Ok(Arc::new(AvxDct4MixedRadix13f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix13f;
            Ok(Arc::new(NeonDct4MixedRadix13f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix13;
            Ok(Arc::new(Dct4MixedRadix13::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix17(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix17f;
            return Ok(Arc::new(AvxDct4MixedRadix17f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix17f;
            Ok(Arc::new(NeonDct4MixedRadix17f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix17;
            Ok(Arc::new(Dct4MixedRadix17::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix19(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix19f;
            return Ok(Arc::new(AvxDct4MixedRadix19f::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix19f;
            Ok(Arc::new(NeonDct4MixedRadix19f::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix19;
            Ok(Arc::new(Dct4MixedRadix19::new(len, half_dct4)?))
        }
    }

    fn dct4_fft(fft: Arc<dyn FftExecutor<Self> + Send + Sync>) -> Returning<Self> {
        Ok(Arc::new(Dct4Fft::new(fft)))
    }

    fn dct4_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Identity::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly2::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly3;
                return Arc::new(AvxDct4Butterfly3::default());
            }
            Arc::new(Dct4Butterfly3::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly4::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly5;
            Arc::new(Dct4Butterfly5::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly6::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly7;
                return Arc::new(AvxDct4Butterfly7::default());
            }
            use crate::dct4::Dct4Butterfly7;
            Arc::new(Dct4Butterfly7::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly8::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly9;
                return Arc::new(AvxDct4Butterfly9::default());
            }
            use crate::dct4::Dct4Butterfly9;
            Arc::new(Dct4Butterfly9::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly10::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly11;
                return Arc::new(AvxDct4Butterfly11::default());
            }
            use crate::dct4::Dct4Butterfly11;
            Arc::new(Dct4Butterfly11::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly12::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly13;
                return Arc::new(AvxDct4Butterfly13::default());
            }
            use crate::dct4::Dct4Butterfly13;
            Arc::new(Dct4Butterfly13::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly14::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly16::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly17;
            Arc::new(Dct4Butterfly17::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly18::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly19;
            Arc::new(Dct4Butterfly19::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly20::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly22::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly23;
            Arc::new(Dct4Butterfly23::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly24::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly26::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct4Butterfly27f;
                Arc::new(NeonDct4Butterfly27f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly27f;
                return Arc::new(AvxDct4Butterfly27f::default());
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::dct4::Dct4Butterfly27;
                Arc::new(Dct4Butterfly27::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly28::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly29;
            Arc::new(Dct4Butterfly29::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly30::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly32::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }
}

impl Dct4Factory for f64 {
    fn dct4_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4Radix2d;
            return Ok(Arc::new(AvxDct4Radix2d::new(len, half_dct2)?));
        }
        Ok(Arc::new(Dct4Radix2::new(len, half_dct2)?))
    }

    fn dct4_mixed_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix2d;
            return Ok(Arc::new(AvxDct4MixedRadix2d::new(len, half_dct2)?));
        }
        use crate::dct4::Dct4MixedRadix2;
        Ok(Arc::new(Dct4MixedRadix2::new(len, half_dct2)?))
    }

    fn dct4_mixed_radix3(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix3d;
            return Ok(Arc::new(AvxDct4MixedRadix3d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix3d;
            Ok(Arc::new(NeonDct4MixedRadix3d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix3;
            Ok(Arc::new(Dct4MixedRadix3::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix5(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix5d;
            return Ok(Arc::new(AvxDct4MixedRadix5d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix5d;
            Ok(Arc::new(NeonDct4MixedRadix5d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix5;
            Ok(Arc::new(Dct4MixedRadix5::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix7(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix7d;
            return Ok(Arc::new(AvxDct4MixedRadix7d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix7d;
            Ok(Arc::new(NeonDct4MixedRadix7d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix7;
            Ok(Arc::new(Dct4MixedRadix7::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix9(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix9d;
            return Ok(Arc::new(AvxDct4MixedRadix9d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix9d;
            Ok(Arc::new(NeonDct4MixedRadix9d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix9;
            Ok(Arc::new(Dct4MixedRadix9::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix11(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix11d;
            return Ok(Arc::new(AvxDct4MixedRadix11d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix11d;
            Ok(Arc::new(NeonDct4MixedRadix11d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix11;
            Ok(Arc::new(Dct4MixedRadix11::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix13(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix13d;
            return Ok(Arc::new(AvxDct4MixedRadix13d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix13d;
            Ok(Arc::new(NeonDct4MixedRadix13d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix13;
            Ok(Arc::new(Dct4MixedRadix13::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix17(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix17d;
            return Ok(Arc::new(AvxDct4MixedRadix17d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix17d;
            Ok(Arc::new(NeonDct4MixedRadix17d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix17;
            Ok(Arc::new(Dct4MixedRadix17::new(len, half_dct4)?))
        }
    }

    fn dct4_mixed_radix19(
        len: usize,
        half_dct4: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct4MixedRadix19d;
            return Ok(Arc::new(AvxDct4MixedRadix19d::new(len, half_dct4)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct4MixedRadix19d;
            Ok(Arc::new(NeonDct4MixedRadix19d::new(len, half_dct4)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::dct4::Dct4MixedRadix19;
            Ok(Arc::new(Dct4MixedRadix19::new(len, half_dct4)?))
        }
    }

    fn dct4_fft(fft: Arc<dyn FftExecutor<Self> + Send + Sync>) -> Returning<Self> {
        Ok(Arc::new(Dct4Fft::new(fft)))
    }

    fn dct4_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Identity::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly2::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly3;
                return Arc::new(AvxDct4Butterfly3::default());
            }
            Arc::new(Dct4Butterfly3::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly4::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly5;
            Arc::new(Dct4Butterfly5::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly6::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly7;
            Arc::new(Dct4Butterfly7::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly8::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly9;
                return Arc::new(AvxDct4Butterfly9::default());
            }
            use crate::dct4::Dct4Butterfly9;
            Arc::new(Dct4Butterfly9::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly10::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly11;
                return Arc::new(AvxDct4Butterfly11::default());
            }
            use crate::dct4::Dct4Butterfly11;
            Arc::new(Dct4Butterfly11::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly12::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::AvxDct4Butterfly13;
                return Arc::new(AvxDct4Butterfly13::default());
            }
            use crate::dct4::Dct4Butterfly13;
            Arc::new(Dct4Butterfly13::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly14::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly16::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly17;
            Arc::new(Dct4Butterfly17::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly18::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly19;
            Arc::new(Dct4Butterfly19::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly20::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly22::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly23;
            Arc::new(Dct4Butterfly23::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly24::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly26::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly27;
            Arc::new(Dct4Butterfly27::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly28::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::dct4::Dct4Butterfly29;
            Arc::new(Dct4Butterfly29::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly30::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct4_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly32::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }
}
