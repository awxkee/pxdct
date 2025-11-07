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
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use pxfm::{f_sincospi, f_sincospif};

pub(crate) trait FftTrigonometry {
    fn sincos_pi(self) -> (Self, Self)
    where
        Self: Sized;
}

impl FftTrigonometry for f32 {
    #[inline]
    fn sincos_pi(self) -> (Self, Self) {
        f_sincospif(self)
    }
}

impl FftTrigonometry for f64 {
    #[inline]
    fn sincos_pi(self) -> (Self, Self) {
        f_sincospi(self)
    }
}

#[inline]
pub(crate) fn compute_twiddle<T: Float + FftTrigonometry + 'static>(
    index: usize,
    fft_len: usize,
) -> Complex<T>
where
    f64: AsPrimitive<T>,
{
    let angle = (-2. * index as f64 / fft_len as f64).as_();
    let (v_sin, v_cos) = angle.sincos_pi();
    Complex {
        re: v_cos,
        im: v_sin,
    }
}
