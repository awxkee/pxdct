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
    Dct4Butterfly30, Dct4Butterfly32, Dct4Fft, Dct4Identity, Dct4MixedRadix2, Dct4Radix2,
};
use crate::factory_dct2::Returning;
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
    fn dct4_fft(fft: Arc<dyn FftExecutor<Self> + Send + Sync>) -> Returning<Self>;
    fn dct4_identity() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly12() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly14() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly16() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly20() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly22() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly24() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly26() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly30() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct4_butterfly32() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dct4Factory for f32 {
    fn dct4_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        Ok(Arc::new(Dct4Radix2::new(len, half_dct2)?))
    }

    fn dct4_mixed_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        Ok(Arc::new(Dct4MixedRadix2::new(len, half_dct2)?))
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

    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly6::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
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

    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly10::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
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

    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly18::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
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

    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly28::default()) as Arc<dyn PxdctExecutor<f32> + Send + Sync>
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
        Ok(Arc::new(Dct4Radix2::new(len, half_dct2)?))
    }

    fn dct4_mixed_radix2(
        len: usize,
        half_dct2: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self> {
        Ok(Arc::new(Dct4MixedRadix2::new(len, half_dct2)?))
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

    fn dct4_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly6::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
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

    fn dct4_butterfly10() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly10::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
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

    fn dct4_butterfly18() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly18::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
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

    fn dct4_butterfly28() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn PxdctExecutor<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            Arc::new(Dct4Butterfly28::default()) as Arc<dyn PxdctExecutor<f64> + Send + Sync>
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
