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
use crate::twiddles::FftTrigonometry;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) fn radixq_odd_twiddlej<T: Float + FftTrigonometry + 'static>(
    q: usize,
    m: usize,
    j: usize,
    k: T,
    dct_len: usize,
) -> T
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let inv_module = q.as_() - 1f64.as_() - 2f64.as_() * m.as_();
    let angle_b = inv_module * k / (2. * dct_len as f64).as_();
    let theta = inv_module * (2 * j).as_();
    let angle_b_phase = theta / q.as_();
    let b_phase = angle_b.cospi();
    let lo = angle_b_phase.sinpi();

    b_phase * lo
}

pub(crate) fn radixq_even_twiddle<T: Float + FftTrigonometry + 'static>(
    q: usize,
    m: usize,
    j: usize,
    k: T,
    fft_len: usize,
) -> T
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let module = (q - 1 - 2 * m).as_();
    let angle_a = module * k / (2. * fft_len as f64).as_();
    let angle_a_phase = 2f64.as_() * module * j.as_() / q.as_();
    let a = angle_a.cospi();
    let a_phase = angle_a_phase.cospi();
    a * a_phase
}

pub(crate) fn radixq_cos_twiddle<T: Float + FftTrigonometry + 'static>(
    q: usize,
    m: usize,
    k: T,
    fft_len: usize,
) -> T
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let module = (q - 1 - 2 * m).as_();
    let angle_a = module * k / (2. * fft_len as f64).as_();
    angle_a.cospi()
}

pub(crate) fn radixq_rotation_twiddle<T: Float + FftTrigonometry + 'static>(
    q: usize,
    m: usize,
    k: T,
    inv_k: T,
    fft_len: usize,
) -> Complex<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let module = (q - 1 - 2 * m).as_();
    let angle_c = module * k / (2. * fft_len as f64).as_();
    let angle_s = module * inv_k / (2. * fft_len as f64).as_();
    let hi = angle_c.tanpi();
    let lo = angle_s.tanpi();
    Complex { re: hi, im: lo }
}
