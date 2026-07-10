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
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum PxdctError {
    OutOfMemory(usize),
    CantCreateUnderlyingFft(String),
    InvalidSizeMultiplier(usize, usize),
    SizeOverflow(usize, usize),
    FftError(String),
    ZeroSizedDct,
    ScratchBufferIsTooSmall(usize, usize),
    InvalidScratchSize(usize, usize),
    OutOfPlaceSizeDoesntMatch(usize, usize),
    MinimumPoints(usize, String),
    OnlyEvenTransform(usize),
}

impl Error for PxdctError {}

impl Display for PxdctError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PxdctError::CantCreateUnderlyingFft(s) => f.write_str(s),
            PxdctError::OutOfMemory(length) => {
                f.write_fmt(format_args!("Cannot allocate {length} bytes to vector",))
            }
            PxdctError::InvalidSizeMultiplier(s0, s1) => f.write_fmt(format_args!(
                "Size {s0} is assumed to be multiplier of {s1} to execute many DFT, but it wasn't"
            )),
            PxdctError::SizeOverflow(a, b) => f.write_fmt(format_args!(
                "Transform size calculation overflowed for {a} and {b}"
            )),
            PxdctError::FftError(s) => f.write_fmt(format_args!("Underlying fft error {s}")),
            PxdctError::ZeroSizedDct => f.write_str("Zero sized DCT is not allowed"),
            PxdctError::ScratchBufferIsTooSmall(a, b) => f.write_fmt(format_args!(
                "Scratch is expected to be at least {a} bytes, but was {b}"
            )),
            PxdctError::InvalidScratchSize(u0, u1) => f.write_fmt(format_args!(
                "Scratch is expected to be at least {u0} bytes, but was {u1}"
            )),
            PxdctError::OutOfPlaceSizeDoesntMatch(u0, u1) => f.write_fmt(format_args!(
                "Input and output sizes doesn't match {u0} vs {u1}"
            )),
            PxdctError::MinimumPoints(points, dct_type) => {
                f.write_fmt(format_args!("{dct_type} requires at least {points} points"))
            }
            PxdctError::OnlyEvenTransform(q) => f.write_fmt(format_args!(
                "Transform function must have even length, but received {q}"
            )),
        }
    }
}
