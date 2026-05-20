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
use crate::type2::Dct2Coprime;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use crate::{PxdctError, PxdctExecutor};
use std::sync::{Arc, OnceLock};

pub(crate) type Returning<T> = Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>;

pub(crate) trait Dct2Factory {
    fn split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix2(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix3(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix5(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix6(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix7(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix9(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix11(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn mixed_radix13(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct2_relatively_prime(
        width_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        height_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly25() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly31() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly37() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly42() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly48() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly49() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly81() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly216() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly243() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dct2Factory for f32 {
    fn split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxSplitRadixDct2f;
            return Ok(Arc::new(AvxSplitRadixDct2f::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonSplitRadixDct2f;
            Ok(
                Arc::new(NeonSplitRadixDct2f::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>,
            )
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::SplitRadixDct2;
            Ok(
                Arc::new(SplitRadixDct2::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>,
            )
        }
    }

    fn mixed_radix2(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix2;
            return Ok(Arc::new(AvxDct2MixedRadix2::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix2;
            Ok(Arc::new(NeonDct2MixedRadix2::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix2;
            Ok(Arc::new(Dct2MixedRadix2::new(length, inner_dct)?))
        }
    }

    fn mixed_radix3(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix3f;
            return Ok(Arc::new(AvxDct2MixedRadix3f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix3f;
            Ok(Arc::new(NeonDct2MixedRadix3f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix3q;
            Ok(Arc::new(Dct2MixedRadix3q::new(length, inner_dct)?))
        }
    }

    fn mixed_radix5(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix5f;
            return Ok(Arc::new(AvxDct2MixedRadix5f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix5f;
            Ok(Arc::new(NeonDct2MixedRadix5f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix5;
            Ok(Arc::new(Dct2MixedRadix5::new(length, inner_dct)?))
        }
    }

    fn mixed_radix6(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix6f;
            return Ok(Arc::new(AvxDct2MixedRadix6f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix6f;
            Ok(Arc::new(NeonDct2MixedRadix6f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix6;
            Ok(Arc::new(Dct2MixedRadix6::new(length, inner_dct)?))
        }
    }

    fn mixed_radix7(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix7f;
            return Ok(Arc::new(AvxDct2MixedRadix7f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix7f;
            Ok(Arc::new(NeonDct2MixedRadix7f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix7;
            Ok(Arc::new(Dct2MixedRadix7::new(length, inner_dct)?))
        }
    }

    fn mixed_radix9(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix9f;
            return Ok(Arc::new(AvxDct2MixedRadix9f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix9f;
            Ok(Arc::new(NeonDct2MixedRadix9f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix9;
            Ok(Arc::new(Dct2MixedRadix9::new(length, inner_dct)?))
        }
    }

    fn mixed_radix11(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix11f;
            return Ok(Arc::new(AvxDct2MixedRadix11f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix11f;
            Ok(Arc::new(NeonDct2MixedRadix11f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix11;
            Ok(Arc::new(Dct2MixedRadix11::new(length, inner_dct)?))
        }
    }

    fn mixed_radix13(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix13f;
            return Ok(Arc::new(AvxDct2MixedRadix13f::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f32> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix13f;
            Ok(Arc::new(NeonDct2MixedRadix13f::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix13;
            Ok(Arc::new(Dct2MixedRadix13::new(length, inner_dct)?))
        }
    }

    fn dct2_relatively_prime(
        width_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        height_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        Ok(Arc::new(Dct2Coprime::new(width_dct, height_dct)?))
    }

    fn dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::type2::power2_butterflies::Dct2Butterfly2;
            Arc::new(Dct2Butterfly2::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::prime_butterflies::Dct2Butterfly3;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly3;
                return Arc::new(AvxDct2Butterfly3::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly3::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly4;
                return Arc::new(AvxDct2Butterfly4::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::power2_butterflies::Dct2Butterfly4;
            Arc::new(Dct2Butterfly4::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly5;
                return Arc::new(AvxDct2Butterfly5::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly5;
            Arc::new(Dct2Butterfly5::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly6;
                return Arc::new(AvxDct2Butterfly6::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::butterflies::Dct2Butterfly6;
            Arc::new(Dct2Butterfly6::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly7;
                return Arc::new(AvxDct2Butterfly7::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly7;
            Arc::new(Dct2Butterfly7::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly8;
                return Arc::new(AvxDct2Butterfly8::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::power2_butterflies::Dct2Butterfly8;
            Arc::new(Dct2Butterfly8::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly9;
                return Arc::new(AvxDct2Butterfly9::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }

            use crate::butterflies::Dct2Butterfly9;
            Arc::new(Dct2Butterfly9::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly10;
            Arc::new(Dct2Butterfly10::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly11;
                return Arc::new(AvxDct2Butterfly11::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly11;
            Arc::new(Dct2Butterfly11::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly12;
                return Arc::new(AvxDct2Butterfly12::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::butterflies::Dct2Butterfly12;
            Arc::new(Dct2Butterfly12::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly13;
                return Arc::new(AvxDct2Butterfly13::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly13;
            Arc::new(Dct2Butterfly13::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly14;
            Arc::new(Dct2Butterfly14::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly15;
            Arc::new(Dct2Butterfly15::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dct2Butterfly16;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly16;
                return Arc::new(AvxDct2Butterfly16::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly16::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly17;
                return Arc::new(AvxDct2Butterfly17::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly17;
            Arc::new(Dct2Butterfly17::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly18;
            Arc::new(Dct2Butterfly18::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly19;
                return Arc::new(AvxDct2Butterfly19::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly19;
            Arc::new(Dct2Butterfly19::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly20;
            Arc::new(Dct2Butterfly20::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly22;
            Arc::new(Dct2Butterfly22::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly23;
                return Arc::new(AvxDct2Butterfly23::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly23;
            Arc::new(Dct2Butterfly23::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly24;
            Arc::new(Dct2Butterfly24::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly25() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly25f;
                return Arc::new(AvxDct2Butterfly25f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly25f;
                Arc::new(NeonDct2Butterfly25f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::Dct2Butterfly25;
                Arc::new(Dct2Butterfly25::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly26;
            Arc::new(Dct2Butterfly26::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly27f;
                return Arc::new(AvxDct2Butterfly27f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly27f;
                Arc::new(NeonDct2Butterfly27f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly27;
                Arc::new(Dct2Butterfly27::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly29;
                return Arc::new(AvxDct2Butterfly29::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly29;
            Arc::new(Dct2Butterfly29::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly30;
            Arc::new(Dct2Butterfly30::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly31() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly31;
                return Arc::new(AvxDct2Butterfly31::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly31;
            Arc::new(Dct2Butterfly31::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly32f;
                return Arc::new(AvxDct2Butterfly32f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly32f;
                Arc::new(NeonDct2Butterfly32f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly32;
                Arc::new(Dct2Butterfly32::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly35;
            Arc::new(Dct2Butterfly35::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly36f;
                return Arc::new(AvxDct2Butterfly36f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly36f;
                Arc::new(NeonDct2Butterfly36f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly36;
                Arc::new(Dct2Butterfly36::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly37() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly37;
                return Arc::new(AvxDct2Butterfly37::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly37;
            Arc::new(Dct2Butterfly37::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly42() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly42;
            Arc::new(Dct2Butterfly42::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly48() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly48;
            Arc::new(Dct2Butterfly48::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly49() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly49f;
                return Arc::new(AvxDct2Butterfly49f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly49f;
                Arc::new(NeonDct2Butterfly49f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::Dct2Butterfly49;
                Arc::new(Dct2Butterfly49::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly64f;
                return Arc::new(AvxDct2Butterfly64f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly64f;
                Arc::new(NeonDct2Butterfly64f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly64;
                Arc::new(Dct2Butterfly64::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly81() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly81f;
                return Arc::new(AvxDct2Butterfly81f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly81f;
                Arc::new(NeonDct2Butterfly81f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly81;
                Arc::new(Dct2Butterfly81::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly128f;
                return Arc::new(AvxDct2Butterfly128f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly128f;
                Arc::new(NeonDct2Butterfly128f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly128;
                Arc::new(Dct2Butterfly128::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly216() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly216f;
                return Arc::new(AvxDct2Butterfly216f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly216f;
                Arc::new(NeonDct2Butterfly216f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly216;
                Arc::new(Dct2Butterfly216::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly243() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly243f;
                return Arc::new(AvxDct2Butterfly243f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly243f;
                Arc::new(NeonDct2Butterfly243f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly243;
                Arc::new(Dct2Butterfly243::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly256f;
                return Arc::new(AvxDct2Butterfly256f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly256f;
                Arc::new(NeonDct2Butterfly256f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly256;
                Arc::new(Dct2Butterfly256::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly512f;
                return Arc::new(AvxDct2Butterfly512f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly512f;
                Arc::new(NeonDct2Butterfly512f::default())
                    as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly512;
                Arc::new(Dct2Butterfly512::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly21;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct2Butterfly21::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
        })
        .clone()
    }
}

impl Dct2Factory for f64 {
    fn split_radix(
        length: usize,
        half_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxSplitRadixDct2d;
            return Ok(Arc::new(AvxSplitRadixDct2d::new(
                length,
                half_dct,
                quarter_dct,
            )?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonSplitRadixDct2d;
            Ok(
                Arc::new(NeonSplitRadixDct2d::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>,
            )
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::SplitRadixDct2;
            Ok(
                Arc::new(SplitRadixDct2::new(length, half_dct, quarter_dct)?)
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>,
            )
        }
    }

    fn mixed_radix2(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f64> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix2;
            return Ok(Arc::new(AvxDct2MixedRadix2::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix2;
            Ok(Arc::new(NeonDct2MixedRadix2::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix2;
            Ok(Arc::new(Dct2MixedRadix2::new(length, inner_dct)?))
        }
    }

    fn mixed_radix3(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f64> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix3d;
            return Ok(Arc::new(AvxDct2MixedRadix3d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix3d;
            Ok(Arc::new(NeonDct2MixedRadix3d::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix3q;
            Ok(Arc::new(Dct2MixedRadix3q::new(length, inner_dct)?))
        }
    }

    fn mixed_radix5(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix5d;
            return Ok(Arc::new(AvxDct2MixedRadix5d::new(length, inner_dct)?));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix5d;
            Ok(Arc::new(NeonDct2MixedRadix5d::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix5;
            Ok(Arc::new(Dct2MixedRadix5::new(length, inner_dct)?))
        }
    }

    fn mixed_radix6(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f64> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix6d;
            return Ok(Arc::new(AvxDct2MixedRadix6d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix6d;
            Ok(Arc::new(NeonDct2MixedRadix6d::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix6;
            Ok(Arc::new(Dct2MixedRadix6::new(length, inner_dct)?))
        }
    }

    fn mixed_radix7(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix7d;
            return Ok(Arc::new(AvxDct2MixedRadix7d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix7d;
            Ok(Arc::new(NeonDct2MixedRadix7d::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix7;
            Ok(Arc::new(Dct2MixedRadix7::new(length, inner_dct)?))
        }
    }

    fn mixed_radix9(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix9d;
            return Ok(Arc::new(AvxDct2MixedRadix9d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonDct2MixedRadix9d;
            Ok(Arc::new(NeonDct2MixedRadix9d::new(length, inner_dct)?))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::type2::Dct2MixedRadix9;
            Ok(Arc::new(Dct2MixedRadix9::new(length, inner_dct)?))
        }
    }

    fn mixed_radix11(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix11d;
            return Ok(Arc::new(AvxDct2MixedRadix11d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        use crate::type2::Dct2MixedRadix11;
        Ok(Arc::new(Dct2MixedRadix11::new(length, inner_dct)?))
    }

    fn mixed_radix13(
        length: usize,
        inner_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxDct2MixedRadix13d;
            return Ok(Arc::new(AvxDct2MixedRadix13d::new(length, inner_dct)?)
                as Arc<dyn PxdctExecutor<f64> + Send + Sync>);
        }
        use crate::type2::Dct2MixedRadix13;
        Ok(Arc::new(Dct2MixedRadix13::new(length, inner_dct)?))
    }

    fn dct2_relatively_prime(
        width_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        height_dct: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        Ok(Arc::new(Dct2Coprime::new(width_dct, height_dct)?))
    }

    fn dct2_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dct2Butterfly2;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct2Butterfly2::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::prime_butterflies::Dct2Butterfly3;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly3;
                return Arc::new(AvxDct2Butterfly3::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly3::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dct2Butterfly4;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly4;
                return Arc::new(AvxDct2Butterfly4::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly4::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly5;
                return Arc::new(AvxDct2Butterfly5::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly5;
            Arc::new(Dct2Butterfly5::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly6;
                return Arc::new(AvxDct2Butterfly6::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::butterflies::Dct2Butterfly6;
            Arc::new(Dct2Butterfly6::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly7;
                return Arc::new(AvxDct2Butterfly7::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly7;
            Arc::new(Dct2Butterfly7::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dct2Butterfly8;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly8;
                return Arc::new(AvxDct2Butterfly8::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly8::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }
    fn dct2_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly9;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly9;
                return Arc::new(AvxDct2Butterfly9::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly9::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly10;
            Arc::new(Dct2Butterfly10::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly11() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly11;
                return Arc::new(AvxDct2Butterfly11::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly11;
            Arc::new(Dct2Butterfly11::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly12;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly12;
                return Arc::new(AvxDct2Butterfly12::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly12::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly13() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly13;
                return Arc::new(AvxDct2Butterfly13::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly13;
            Arc::new(Dct2Butterfly13::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly14;
            Arc::new(Dct2Butterfly14::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly15() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly15;
            Arc::new(Dct2Butterfly15::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type2::power2_butterflies::Dct2Butterfly16;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly16;
                return Arc::new(AvxDct2Butterfly16::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly16::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly17;
                return Arc::new(AvxDct2Butterfly17::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly17;
            Arc::new(Dct2Butterfly17::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly18;
            Arc::new(Dct2Butterfly18::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly19() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly19;
                return Arc::new(AvxDct2Butterfly19::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly19;
            Arc::new(Dct2Butterfly19::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly20;
            Arc::new(Dct2Butterfly20::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly22;
            Arc::new(Dct2Butterfly22::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly23() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly23;
                return Arc::new(AvxDct2Butterfly23::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly23;
            Arc::new(Dct2Butterfly23::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly24;
            Arc::new(Dct2Butterfly24::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly25() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly25d;
                return Arc::new(AvxDct2Butterfly25d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::Dct2Butterfly25;
            Arc::new(Dct2Butterfly25::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly26;
            Arc::new(Dct2Butterfly26::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly27() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly27;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly27d;
                return Arc::new(AvxDct2Butterfly27d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly27::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly29() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly29;
                return Arc::new(AvxDct2Butterfly29::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly29;
            Arc::new(Dct2Butterfly29::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly30;
            Arc::new(Dct2Butterfly30::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly31() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly31;
                return Arc::new(AvxDct2Butterfly31::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly31;
            Arc::new(Dct2Butterfly31::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly32d;
                return Arc::new(AvxDct2Butterfly32d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly32d;
                Arc::new(NeonDct2Butterfly32d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly32;
                Arc::new(Dct2Butterfly32::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly35() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly35;
            Arc::new(Dct2Butterfly35::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly36() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly36d;
                Arc::new(NeonDct2Butterfly36d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly36d;
                return Arc::new(AvxDct2Butterfly36d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly36;
                Arc::new(Dct2Butterfly36::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly37() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly37;
                return Arc::new(AvxDct2Butterfly37::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::prime_butterflies::Dct2Butterfly37;
            Arc::new(Dct2Butterfly37::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly42() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly42;
            Arc::new(Dct2Butterfly42::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly48() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::Dct2Butterfly48;
            Arc::new(Dct2Butterfly48::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly49() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly49d;
                return Arc::new(AvxDct2Butterfly49d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::type2::Dct2Butterfly49;
            Arc::new(Dct2Butterfly49::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly64() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly64d;
                return Arc::new(AvxDct2Butterfly64d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly64d;
                Arc::new(NeonDct2Butterfly64d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly64;
                Arc::new(Dct2Butterfly64::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly81() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly81;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly81d;
                return Arc::new(AvxDct2Butterfly81d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            Arc::new(Dct2Butterfly81::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly128() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly128d;
                return Arc::new(AvxDct2Butterfly128d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly128d;
                Arc::new(NeonDct2Butterfly128d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly128;
                Arc::new(Dct2Butterfly128::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly216() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly216d;
                Arc::new(NeonDct2Butterfly216d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly216d;
                return Arc::new(AvxDct2Butterfly216d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::Dct2Butterfly216;
                Arc::new(Dct2Butterfly216::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly243() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly243d;
                return Arc::new(AvxDct2Butterfly243d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            use crate::butterflies::Dct2Butterfly243;
            Arc::new(Dct2Butterfly243::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }

    fn dct2_butterfly256() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly256d;
                return Arc::new(AvxDct2Butterfly256d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly256d;
                Arc::new(NeonDct2Butterfly256d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly256;
                Arc::new(Dct2Butterfly256::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly512() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxDct2Butterfly512d;
                return Arc::new(AvxDct2Butterfly512d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::NeonDct2Butterfly512d;
                Arc::new(NeonDct2Butterfly512d::default())
                    as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::type2::power2_butterflies::Dct2Butterfly512;
                Arc::new(Dct2Butterfly512::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
            }
        })
        .clone()
    }

    fn dct2_butterfly21() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::butterflies::Dct2Butterfly21;
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct2Butterfly21::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
        })
        .clone()
    }
}
