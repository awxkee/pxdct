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
use crate::SpectralExecutor;
use crate::factory_dct2::Returning;
use std::sync::Arc;

pub(crate) trait Dst7Factory {
    fn dst7_fft(length: usize) -> Returning<Self>;
    fn dst7_butterfly2() -> SpectralExecutor<Self>;
    fn dst7_butterfly3() -> SpectralExecutor<Self>;
    fn dst7_butterfly4() -> SpectralExecutor<Self>;
    fn dst7_butterfly5() -> SpectralExecutor<Self>;
    fn dst7_butterfly6() -> SpectralExecutor<Self>;
    fn dst7_butterfly7() -> SpectralExecutor<Self>;
    fn dst7_butterfly8() -> SpectralExecutor<Self>;
    fn dst7_butterfly16() -> SpectralExecutor<Self>;
}

macro_rules! define_factory {
    ($for_type: ident) => {
        impl Dst7Factory for $for_type {
            fn dst7_fft(length: usize) -> Returning<Self> {
                use crate::type7::Dst7Fft;
                Ok(Arc::new(Dst7Fft::new(length)?))
            }

            fn dst7_butterfly2() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly2;
                Arc::new(Dst7Butterfly2::default())
            }

            fn dst7_butterfly3() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly3;
                Arc::new(Dst7Butterfly3::default())
            }

            fn dst7_butterfly4() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly4;
                Arc::new(Dst7Butterfly4::default())
            }

            fn dst7_butterfly5() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly5;
                Arc::new(Dst7Butterfly5::default())
            }

            fn dst7_butterfly6() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly6;
                Arc::new(Dst7Butterfly6::default())
            }

            fn dst7_butterfly7() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly7;
                Arc::new(Dst7Butterfly7::default())
            }

            fn dst7_butterfly8() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly8;
                Arc::new(Dst7Butterfly8::default())
            }

            fn dst7_butterfly16() -> SpectralExecutor<Self> {
                use crate::type7::Dst7Butterfly16;
                Arc::new(Dst7Butterfly16::default())
            }
        }
    };
}

define_factory!(f32);
define_factory!(f64);
