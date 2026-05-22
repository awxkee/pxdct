/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
#![allow(clippy::too_many_arguments)]
mod dct2_bf_power2;
mod dct2_bf_radix3;
mod dct2_butterflies;
mod mul_f32;
mod pfa_dct2_remap;
mod stored;
mod storef;
mod transpose_real_s;
mod type2;
mod type3;
mod type4;
mod util;

pub(crate) use dct2_bf_power2::{
    AvxDct2Butterfly4, AvxDct2Butterfly8, AvxDct2Butterfly16, AvxDst2Butterfly4,
};
pub(crate) use dct2_bf_radix3::{
    AvxDct2Butterfly9, AvxDct2Butterfly27f, AvxDct2Butterfly81f, AvxDct2Butterfly243f,
};
pub(crate) use dct2_butterflies::AvxDct2Butterfly12;
pub(crate) use mul_f32::AvxDctSpectrumMulF32;
#[cfg(target_pointer_width = "64")]
pub(crate) use pfa_dct2_remap::AvxPfaDct2Remapper;
pub(crate) use transpose_real_s::AvxTransposeFReal4x4;
pub(crate) use type2::{
    AvxDct2Butterfly3, AvxDct2Butterfly5, AvxDct2Butterfly6, AvxDct2Butterfly7, AvxDct2Butterfly11,
    AvxDct2Butterfly13, AvxDct2Butterfly17, AvxDct2Butterfly19, AvxDct2Butterfly23,
    AvxDct2Butterfly25d, AvxDct2Butterfly25f, AvxDct2Butterfly27d, AvxDct2Butterfly29,
    AvxDct2Butterfly31, AvxDct2Butterfly32d, AvxDct2Butterfly32f, AvxDct2Butterfly36d,
    AvxDct2Butterfly36f, AvxDct2Butterfly37, AvxDct2Butterfly49d, AvxDct2Butterfly49f,
    AvxDct2Butterfly64d, AvxDct2Butterfly64f, AvxDct2Butterfly81d, AvxDct2Butterfly128d,
    AvxDct2Butterfly128f, AvxDct2Butterfly216d, AvxDct2Butterfly216f, AvxDct2Butterfly243d,
    AvxDct2Butterfly256d, AvxDct2Butterfly256f, AvxDct2Butterfly512d, AvxDct2Butterfly512f,
    AvxDct2MixedRadix2, AvxDct2MixedRadix3d, AvxDct2MixedRadix3f, AvxDct2MixedRadix5d,
    AvxDct2MixedRadix5f, AvxDct2MixedRadix6d, AvxDct2MixedRadix6f, AvxDct2MixedRadix7d,
    AvxDct2MixedRadix7f, AvxDct2MixedRadix9d, AvxDct2MixedRadix9f, AvxDct2MixedRadix11d,
    AvxDct2MixedRadix11f, AvxDct2MixedRadix13d, AvxDct2MixedRadix13f, AvxSplitRadixDct2d,
    AvxSplitRadixDct2f, AvxSplitRadixDst2d, AvxSplitRadixDst2f,
};
pub(crate) use type3::{
    AvxDct3Butterfly16, AvxDct3Butterfly32, AvxDct3Butterfly64, AvxDct3MixedRadix3d,
    AvxDct3MixedRadix3f, AvxDct3MixedRadix5d, AvxDct3MixedRadix5f, AvxDct3MixedRadix7d,
    AvxDct3MixedRadix7f, AvxDct3MixedRadix9d, AvxDct3MixedRadix9f, AvxSplitRadixDct3d,
    AvxSplitRadixDct3f,
};
pub(crate) use type4::{
    AvxDct4Butterfly3, AvxDct4Butterfly7, AvxDct4Butterfly9, AvxDct4Butterfly11,
    AvxDct4Butterfly13, AvxDct4Butterfly27f, AvxDct4MixedRadix2d, AvxDct4MixedRadix2f,
    AvxDct4MixedRadix3d, AvxDct4MixedRadix3f, AvxDct4MixedRadix5d, AvxDct4MixedRadix5f,
    AvxDct4MixedRadix7d, AvxDct4MixedRadix7f, AvxDct4MixedRadix9d, AvxDct4MixedRadix9f,
    AvxDct4MixedRadix11d, AvxDct4MixedRadix11f, AvxDct4MixedRadix13d, AvxDct4MixedRadix13f,
    AvxDct4MixedRadix17d, AvxDct4MixedRadix17f, AvxDct4MixedRadix19d, AvxDct4MixedRadix19f,
    AvxDct4Radix2d, AvxDct4Radix2f,
};
