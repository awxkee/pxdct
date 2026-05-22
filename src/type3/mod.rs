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
mod bf_f2;
mod butterflies;
pub mod fft;
mod identity;
mod mixed_radix3;
mod mixed_radix5;
mod mixed_radix7;
mod mixed_radix9;
mod prime_butterflies;
mod split_radix;

pub(crate) use bf_f2::{
    Dct3Butterfly2, Dct3Butterfly4, Dct3Butterfly8, Dct3Butterfly16, Dct3Butterfly32,
    Dct3Butterfly64,
};
pub(crate) use butterflies::{
    Dct3Butterfly6, Dct3Butterfly9, Dct3Butterfly10, Dct3Butterfly12, Dct3Butterfly14,
    Dct3Butterfly15, Dct3Butterfly18, Dct3Butterfly20, Dct3Butterfly21, Dct3Butterfly24,
    Dct3Butterfly26, Dct3Butterfly28, Dct3Butterfly30, Dct3Butterfly35, Dct3Butterfly36,
};
pub(crate) use fft::Dct3Fft;
pub(crate) use identity::Dct3Identity;
#[allow(unused_imports)]
pub(crate) use mixed_radix3::{Dct3MixedRadix3, radixq_dct3_n_rotation_twiddle};
#[allow(unused_imports)]
pub(crate) use mixed_radix5::{Dct3MixedRadix5, Dct3MixedRadix5Sample};
#[allow(unused_imports)]
pub(crate) use mixed_radix7::{Dct3MixedRadix7, Dct3MixedRadix7Sample};
#[allow(unused_imports)]
pub(crate) use mixed_radix9::{Dct3MixedRadix9, Dct3MixedRadix9Sample};
pub(crate) use prime_butterflies::{
    Dct3Butterfly3, Dct3Butterfly5, Dct3Butterfly7, Dct3Butterfly11, Dct3Butterfly13,
};
#[allow(unused_imports)]
pub(crate) use split_radix::{SplitRadixDct3, SplitRadixDst3};
