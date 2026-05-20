/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

pub(crate) trait Dct1Factory {
    fn split_radix(
        length: usize,
        half_p1_dct1: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        half_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<Self>;
    fn dct1_fft(length: usize) -> Returning<Self>;
    fn dct1_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct1_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

impl Dct1Factory for f32 {
    fn split_radix(
        length: usize,
        half_p1_dct1: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        half_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f32> {
        use crate::type1::SplitRadixDct1;
        Ok(Arc::new(SplitRadixDct1::new(
            length,
            half_p1_dct1,
            half_dct3,
        )?))
    }

    fn dct1_fft(length: usize) -> Returning<Self> {
        use crate::type1::Dct1Fft;
        Ok(Arc::new(Dct1Fft::new(length)?))
    }

    fn dct1_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly2;
        Arc::new(Dct1Butterfly2::default())
    }

    fn dct1_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly3;
        Arc::new(Dct1Butterfly3::default())
    }

    fn dct1_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly4;
        Arc::new(Dct1Butterfly4::default())
    }

    fn dct1_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly5;
        Arc::new(Dct1Butterfly5::default())
    }

    fn dct1_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly6;
        Arc::new(Dct1Butterfly6::default())
    }

    fn dct1_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly7;
        Arc::new(Dct1Butterfly7::default())
    }

    fn dct1_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly8;
        Arc::new(Dct1Butterfly8::default())
    }

    fn dct1_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly9;
        Arc::new(Dct1Butterfly9::default())
    }

    fn dct1_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly17;
        Arc::new(Dct1Butterfly17::default())
    }
}

impl Dct1Factory for f64 {
    fn split_radix(
        length: usize,
        half_p1_dct1: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
        half_dct3: Arc<dyn PxdctExecutor<Self> + Send + Sync>,
    ) -> Returning<f64> {
        use crate::type1::SplitRadixDct1;
        Ok(Arc::new(SplitRadixDct1::new(
            length,
            half_p1_dct1,
            half_dct3,
        )?))
    }

    fn dct1_fft(length: usize) -> Returning<Self> {
        use crate::type1::Dct1Fft;
        Ok(Arc::new(Dct1Fft::new(length)?))
    }

    fn dct1_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly2;
        Arc::new(Dct1Butterfly2::default())
    }

    fn dct1_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly3;
        Arc::new(Dct1Butterfly3::default())
    }

    fn dct1_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly4;
        Arc::new(Dct1Butterfly4::default())
    }

    fn dct1_butterfly5() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly5;
        Arc::new(Dct1Butterfly5::default())
    }

    fn dct1_butterfly6() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly6;
        Arc::new(Dct1Butterfly6::default())
    }

    fn dct1_butterfly7() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly7;
        Arc::new(Dct1Butterfly7::default())
    }

    fn dct1_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly8;
        Arc::new(Dct1Butterfly8::default())
    }

    fn dct1_butterfly9() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly9;
        Arc::new(Dct1Butterfly9::default())
    }

    fn dct1_butterfly17() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
        use crate::type1::Dct1Butterfly17;
        Arc::new(Dct1Butterfly17::default())
    }
}
