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
mod block_transpose;
#[cfg(feature = "fcma")]
mod fcma_mul_f32;
mod mul_f32;
#[cfg(target_pointer_width = "64")]
mod pfa_dct2_remap;
mod store_d;
mod transpose;
mod type2;
mod type3;
mod type4;
mod util;

#[cfg(feature = "fcma")]
pub(crate) use fcma_mul_f32::FcmaDctSpectrumMulF32;
pub(crate) use mul_f32::DctSpectrumMulF32;
pub(crate) use type2::bf_radix3::{
    NeonDct2Butterfly27f, NeonDct2Butterfly81f, NeonDct2Butterfly243f,
};

pub(crate) use block_transpose::{
    NeonTransposeNx5F32, NeonTransposeNx6F32, NeonTransposeNx7F32, NeonTransposeNx11F32,
};
#[cfg(target_pointer_width = "64")]
pub(crate) use pfa_dct2_remap::NeonPfaDct2Remapper;
pub(crate) use transpose::NeonTranspose4x4;
pub(crate) use type2::{
    NeonDct2Butterfly25f, NeonDct2Butterfly32d, NeonDct2Butterfly32f, NeonDct2Butterfly36d,
    NeonDct2Butterfly36f, NeonDct2Butterfly49f, NeonDct2Butterfly64d, NeonDct2Butterfly64f,
    NeonDct2Butterfly128d, NeonDct2Butterfly128f, NeonDct2Butterfly216d, NeonDct2Butterfly216f,
    NeonDct2Butterfly256d, NeonDct2Butterfly256f, NeonDct2Butterfly512d, NeonDct2Butterfly512f,
    NeonDct2MixedRadix2, NeonDct2MixedRadix3d, NeonDct2MixedRadix3f, NeonDct2MixedRadix5d,
    NeonDct2MixedRadix5f, NeonDct2MixedRadix6d, NeonDct2MixedRadix6f, NeonDct2MixedRadix7d,
    NeonDct2MixedRadix7f, NeonDct2MixedRadix9d, NeonDct2MixedRadix9f, NeonDct2MixedRadix11f,
    NeonDct2MixedRadix13f, NeonSplitRadixDct2d, NeonSplitRadixDct2f, NeonSplitRadixDst2d,
    NeonSplitRadixDst2f, ScaledNeonDct2Butterfly32f, ScaledNeonDct2Butterfly64f,
    ScaledNeonDct2Butterfly128f, ScaledNeonDct2Butterfly256f, ScaledNeonDct2Butterfly512f,
    ScaledNeonSplitRadixDct2d, ScaledNeonSplitRadixDct2f,
};
pub(crate) use type3::{
    NeonDct3MixedRadix3d, NeonDct3MixedRadix3f, NeonDct3MixedRadix5d, NeonDct3MixedRadix5f,
    NeonDct3MixedRadix7d, NeonDct3MixedRadix7f, NeonDct3MixedRadix9d, NeonDct3MixedRadix9f,
};
pub(crate) use type3::{NeonSplitRadixDct3d, NeonSplitRadixDct3f};
pub(crate) use type4::{
    NeonDct4Butterfly27f, NeonDct4MixedRadix2f, NeonDct4MixedRadix3d, NeonDct4MixedRadix3f,
    NeonDct4MixedRadix5d, NeonDct4MixedRadix5f, NeonDct4MixedRadix7d, NeonDct4MixedRadix7f,
    NeonDct4MixedRadix9d, NeonDct4MixedRadix9f, NeonDct4MixedRadix11d, NeonDct4MixedRadix11f,
    NeonDct4MixedRadix13d, NeonDct4MixedRadix13f, NeonDct4MixedRadix17d, NeonDct4MixedRadix17f,
    NeonDct4MixedRadix19d, NeonDct4MixedRadix19f, NeonDct4Radix2f,
};
