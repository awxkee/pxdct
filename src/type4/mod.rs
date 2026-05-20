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
mod bf_3n;
mod butterflies;
mod fft;
mod fft_even;
mod identity;
mod mixed_radix2;
mod power2_bf;
mod prime_butterflies;
mod radix11;
mod radix13;
mod radix17;
mod radix19;
mod radix2;
mod radix3;
mod radix5;
mod radix7;
mod radix9;
mod utils;
mod dst;

#[allow(unused)]
pub(crate) use bf_3n::{Dct4Butterfly9, Dct4Butterfly27, Dct4Butterfly81, Dct4MixedRadix9Sample};
pub(crate) use butterflies::{
    Dct4Butterfly6, Dct4Butterfly10, Dct4Butterfly12, Dct4Butterfly14, Dct4Butterfly18,
    Dct4Butterfly20, Dct4Butterfly22, Dct4Butterfly24, Dct4Butterfly26, Dct4Butterfly28,
    Dct4Butterfly30,
};
pub(crate) use fft::Dct4Fft;
pub(crate) use fft_even::Dct4FftEven;
pub(crate) use identity::Dct4Identity;
pub(crate) use mixed_radix2::Dct4MixedRadix2;
pub(crate) use power2_bf::{
    Dct4Butterfly2, Dct4Butterfly4, Dct4Butterfly8, Dct4Butterfly16, Dct4Butterfly32,
};
#[allow(unused)]
pub(crate) use prime_butterflies::{
    Dct4Butterfly3, Dct4Butterfly5, Dct4Butterfly7, Dct4Butterfly11, Dct4Butterfly13,
    Dct4Butterfly17, Dct4Butterfly19, Dct4Butterfly23, Dct4Butterfly29, Dct4MixedRadix5Sample,
    Dct4MixedRadix7Sample, Dct4MixedRadix11Sample, Dct4MixedRadix13Sample, Dct4MixedRadix17Sample,
    Dct4MixedRadix19Sample,
};
pub(crate) use radix2::Dct4Radix2;
#[allow(unused)]
pub(crate) use radix3::Dct4MixedRadix3;
#[allow(unused)]
pub(crate) use radix5::Dct4MixedRadix5;
#[allow(unused)]
pub(crate) use radix7::Dct4MixedRadix7;
#[allow(unused)]
pub(crate) use radix9::Dct4MixedRadix9;
#[allow(unused)]
pub(crate) use radix11::Dct4MixedRadix11;
#[allow(unused)]
pub(crate) use radix13::Dct4MixedRadix13;
#[allow(unused)]
pub(crate) use radix17::Dct4MixedRadix17;
#[allow(unused)]
pub(crate) use radix19::Dct4MixedRadix19;
pub(crate) use utils::radixq_dct4_rotation_twiddle;
