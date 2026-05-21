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

pub(crate) trait Dct7Factory {
    fn dct7_fft(length: usize) -> Returning<Self>;
    fn dct7_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct7_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct7_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
    fn dct7_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync>;
}

macro_rules! define_factory {
    ($for_type: ident) => {
        impl Dct7Factory for $for_type {
            fn dct7_fft(length: usize) -> Returning<Self> {
                use crate::type7::Dct7Fft;
                Ok(Arc::new(Dct7Fft::new(length)?))
            }

            fn dct7_butterfly2() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
                use crate::type7::Dct7Butterfly2;
                Arc::new(Dct7Butterfly2::default())
            }

            fn dct7_butterfly3() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
                use crate::type7::Dct7Butterfly3;
                Arc::new(Dct7Butterfly3::default())
            }

            fn dct7_butterfly4() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
                use crate::type7::Dct7Butterfly4;
                Arc::new(Dct7Butterfly4::default())
            }

            fn dct7_butterfly8() -> Arc<dyn PxdctExecutor<Self> + Send + Sync> {
                use crate::type7::Dct7Butterfly8;
                Arc::new(Dct7Butterfly8::default())
            }
        }
    };
}

define_factory!(f32);
define_factory!(f64);
