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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::type2::power2_butterflies::{Dct2Butterfly4, Dct2Butterfly8};
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dst2Butterfly3<T> {
    _phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dst2Butterfly3<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];

        let sqrt3_div2 = T::SQRT_3 * T::HALF;

        let common_half = (x0 + x2) * T::HALF;

        let y0 = common_half + x1;
        let y1 = (x0 - x2) * sqrt3_div2;
        let y2 = x0 - x1 + x2;

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
    }
}

define_in_place_butterfly!(Dst2Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly5<T: DctSample> {
    /// twiddle0 = compute_twiddle(1, 20).conj()
    ///   re = cos(π/10)  = D = √(10+2√5)/4
    ///   im = sin(π/10)  = A = (√5−1)/4
    twiddle0: Complex<T>,
    /// twiddle1 = compute_twiddle(3, 20).conj()
    ///   re = cos(3π/10) = C = √(10−2√5)/4
    ///   im = sin(3π/10) = B = (√5+1)/4
    twiddle1: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 20).conj(),
            twiddle1: compute_twiddle(3, 20).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly5<T> {
    /// DST-II butterfly for N=5.
    ///
    /// Sum/difference factoring reduces the 5×5 matrix to 8 multiplies:
    ///
    ///   s04 = x0+x4,  d04 = x0-x4
    ///   s13 = x1+x3,  d13 = x1-x3
    ///
    ///   Y[0] = A·s04 + B·s13 + x2
    ///   Y[1] = C·d04 + D·d13
    ///   Y[2] = B·s04 + A·s13 − x2
    ///   Y[3] = D·d04 − C·d13
    ///   Y[4] =   s04 −   s13 + x2
    ///
    /// Constants read directly from twiddle fields:
    ///   A = twiddle0.im,  B = twiddle1.im
    ///   C = twiddle1.re,  D = twiddle0.re
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];

        let s04 = x0 + x4;
        let d04 = x0 - x4;
        let s13 = x1 + x3;
        let d13 = x1 - x3;

        let a = self.twiddle0.im; // sin(π/10)
        let b = self.twiddle1.im; // sin(3π/10)
        let c = self.twiddle1.re; // cos(3π/10)
        let d = self.twiddle0.re; // cos(π/10)

        let a_s04 = a * s04;
        let b_s04 = b * s04;
        let a_s13 = a * s13;
        let b_s13 = b * s13;

        let c_d04 = c * d04;
        let d_d04 = d * d04;
        let c_d13 = c * d13;
        let d_d13 = d * d13;

        data[0] = a_s04 + b_s13 + x2;
        data[1] = c_d04 + d_d13;
        data[2] = b_s04 + a_s13 - x2;
        data[3] = d_d04 - c_d13;
        data[4] = s04 - s13 + x2;
    }
}

define_in_place_butterfly!(Dst2Butterfly5, 5);

// ── N=6 ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly6<T: DctSample> {
    /// twiddle0 = compute_twiddle(1, 24).conj()
    ///   re = cos(π/12)
    ///   im = sin(π/12)
    twiddle0: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 24).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly6<T> {
    /// DST-II butterfly for N=6.
    ///
    /// Decomposes into:
    ///   s[i] = x[i] + x[5-i]   (symmetric pairs)
    ///   d[i] = x[i] - x[5-i]   (antisymmetric pairs)
    ///
    ///   DST-IV(3)(s)  → even outputs Y[0], Y[2], Y[4]
    ///   DST-II(3)(d)  → odd  outputs Y[1], Y[3], Y[5]
    ///
    /// DST-II(3)(d) reuses the Dst2Butterfly3 formulae inline:
    ///   Y[1] = (d0+d2)·½ + d1
    ///   Y[3] = (d0−d2)·(√3/2)
    ///   Y[5] = d0 − d1 + d2
    ///
    /// DST-IV(3)(s) uses one twiddle (fft_len=24):
    ///   s1c = (√2/2)·s1  — hoisted, shared by all three even outputs
    ///   Y[0] = t0.im·s0 + s1c        + t0.re·s2
    ///   Y[2] = (√2/2)·(s0 − s2)     + s1c
    ///   Y[4] = t0.re·s0 − s1c        + t0.im·s2
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];

        let s0 = x0 + x5;
        let s1 = x1 + x4;
        let s2 = x2 + x3;
        let d0 = x0 - x5;
        let d1 = x1 - x4;
        let d2 = x2 - x3;

        let t0 = self.twiddle0; // re=cos(π/12), im=sin(π/12)

        // --- Even outputs: DST-IV(3)(s) ---
        // Hoist s1·(√2/2) — shared by Y[0], Y[2], Y[4]
        let s1c = T::FRAC_1_SQRT_2 * s1;

        data[0] = fmla(t0.re, s2, fmla(t0.im, s0, s1c));
        data[2] = fmla(T::FRAC_1_SQRT_2, s0 - s2, s1c);
        data[4] = fmla(t0.im, s2, fmla(t0.re, s0, -s1c));

        // --- Odd outputs: DST-II(3)(d) ---
        let sqrt3_half = T::SQRT_3 * T::HALF;
        let half_sum = (d0 + d2) * T::HALF;

        data[1] = half_sum + d1;
        data[3] = (d0 - d2) * sqrt3_half;
        data[5] = d0 - d1 + d2;
    }
}

define_in_place_butterfly!(Dst2Butterfly6, 6);

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly7<T: DctSample> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 28).conj(),
            twiddle1: compute_twiddle(2, 28).conj(),
            twiddle2: compute_twiddle(3, 28).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly7<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];

        let s0 = x0 + x6;
        let s1 = x1 + x5;
        let s2 = x2 + x4;
        let d0 = x0 - x6;
        let d1 = x1 - x5;
        let d2 = x2 - x4;

        let t0 = self.twiddle0; // re=cos(π/14),  im=sin(π/14)
        let t1 = self.twiddle1; // re=cos(π/7),   im=sin(π/7)
        let t2 = self.twiddle2; // re=cos(3π/14), im=sin(3π/14)

        // --- Even outputs ---
        data[0] = fmla(t1.re, s2, fmla(t2.im, s1, fmla(t0.im, s0, x3)));
        data[2] = fmla(-t0.im, s2, fmla(t1.re, s1, fmla(t2.im, s0, -x3)));
        data[4] = fmla(-t2.im, s2, fmla(-t0.im, s1, fmla(t1.re, s0, x3)));
        data[6] = s0 - s1 + s2 - x3;

        // --- Odd outputs ---
        data[1] = fmla(t2.re, d2, fmla(t0.re, d1, fmla(t1.im, d0, T::zero())));
        data[3] = fmla(-t0.re, d2, fmla(t1.im, d1, fmla(t2.re, d0, T::zero())));
        data[5] = fmla(t1.im, d2, fmla(-t2.re, d1, fmla(t0.re, d0, T::zero())));
    }
}

define_in_place_butterfly!(Dst2Butterfly7, 7);

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly8<T: DctSample> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 16).conj(),
            twiddle1: compute_twiddle(1, 32).conj(),
            twiddle2: compute_twiddle(3, 32).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly8<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];

        let u0 = x0 + x7;
        let u1 = x1 + x6;
        let u2 = x2 + x5;
        let u3 = x3 + x4;
        let v0 = x0 - x7;
        let v1 = x1 - x6;
        let v2 = x2 - x5;
        let v3 = x3 - x4;

        let p = v0 + v3;
        let q = v1 + v2;
        let dp = v0 - v3;
        let dq = v1 - v2;

        let t0 = self.twiddle0;
        let ap = t0.im * p;
        let bq = t0.re * q;
        let bp = t0.re * p;
        let aq = t0.im * q;

        data[1] = ap + bq;
        data[3] = T::FRAC_1_SQRT_2 * (dp + dq);
        data[5] = bp - aq;
        data[7] = dp - dq;

        let t1 = self.twiddle1;
        let t2 = self.twiddle2;

        let d = t1.im; // sin(π/16)
        let e = t1.re; // cos(π/16)
        let f = t2.im; // sin(3π/16)
        let g = t2.re; // cos(3π/16)

        data[0] = fmla(e, u3, fmla(g, u2, fmla(f, u1, d * u0)));
        data[2] = fmla(-g, u3, fmla(d, u2, fmla(e, u1, f * u0)));
        data[4] = fmla(f, u3, fmla(-e, u2, fmla(d, u1, g * u0)));
        data[6] = fmla(-d, u3, fmla(f, u2, fmla(-g, u1, e * u0)));
    }
}

define_in_place_butterfly!(Dst2Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly9<T: DctSample> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 36).conj(),
            twiddle1: compute_twiddle(2, 36).conj(),
            twiddle2: compute_twiddle(3, 36).conj(),
            twiddle3: compute_twiddle(4, 36).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly9<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];

        let s0 = x0 + x8;
        let s1 = x1 + x7;
        let s2 = x2 + x6;
        let s3 = x3 + x5;

        let d0 = x0 - x8;
        let d1 = x1 - x7;
        let d2 = x2 - x6;
        let d3 = x3 - x5;

        let t0 = self.twiddle0; // re=cos(π/18),  im=sin(π/18)
        let t1 = self.twiddle1; // re=cos(2π/18), im=sin(2π/18)
        let t2 = self.twiddle2; // re=√3/2,       im=1/2
        let t3 = self.twiddle3; // re=cos(4π/18), im=sin(4π/18)

        let s1_t2im = s1 * t2.im;
        let d1_t2re = d1 * t2.re;

        data[0] = fmla(s3, t1.re, fmla(s2, t3.re, fmla(s0, t0.im, s1_t2im)) + x4);
        data[2] = fmla(t2.im, s0 + s2 - s3, s1 - x4);
        data[4] = fmla(-s3, t0.im, fmla(-s2, t1.re, fmla(s0, t3.re, s1_t2im)) + x4);
        data[6] = fmla(s3, t3.re, fmla(-s2, t0.im, fmla(s0, t1.re, -s1_t2im)) - x4);
        data[8] = s0 - s1 + s2 - s3 + x4;
        data[1] = fmla(d3, t3.im, fmla(d2, t0.re, fmla(d0, t1.im, d1_t2re)));
        data[3] = fmla(-d3, t0.re, fmla(-d2, t1.im, fmla(d0, t3.im, d1_t2re)));
        data[5] = t2.re * (d0 - d2 + d3);
        data[7] = fmla(-d3, t1.im, fmla(d2, t3.im, fmla(d0, t0.re, -d1_t2re)));
    }
}

define_in_place_butterfly!(Dst2Butterfly9, 9);

/// Hardcoded split-radix DST-II butterfly for N=16.
///
/// Equivalent to `SplitRadixDst2` wrapping `SplitRadixDct2Impl` with
/// `Dct2Butterfly8` (half) and `Dct2Butterfly4` (quarter), but with:
///   - concrete sub-transform types (no `Arc<dyn>`, no dynamic dispatch)
///   - pre-negation of odd inputs and post-reversal of output fused into
///     the split-radix scatter/gather, eliminating two passes over the data
///   - fixed-size stack buffers replacing scratch allocation
///
/// Algorithm
/// ─────────
/// DST-II(16)(x)[k] = DCT-II(16)(y)[15-k]
///   where y[n] = (−1)^n · x[n]
///
/// The split-radix DCT-II(16) decomposes y into:
///   half_dct  : DCT-II(8) of 8 symmetric sums  (→ even-indexed outputs)
///   quarter_dct: two DCT-II(4) transforms
///                  – one for 4 twiddle-rotated cosine inputs
///                  – one for 4 twiddle-rotated sine inputs (stored reversed
///                    with alternating sign)  (→ odd-indexed outputs)
///
/// Pre-negation of odd x[n] is folded into the sum/difference reads so no
/// separate pass over the data is needed. The output reversal (k → 15−k)
/// is folded into the final scatter step.
///
/// Twiddles: `compute_twiddle(2*i+1, 64).conj()` for i=0..3
///   (same formula as `SplitRadixDct2Impl` for `len=16`)
#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly16<T: DctSample> {
    /// compute_twiddle(1, 64).conj(): re=cos(π/32),  im=sin(π/32)
    twiddle0: Complex<T>,
    /// compute_twiddle(3, 64).conj(): re=cos(3π/32), im=sin(3π/32)
    twiddle1: Complex<T>,
    /// compute_twiddle(5, 64).conj(): re=cos(5π/32), im=sin(5π/32)
    twiddle2: Complex<T>,
    /// compute_twiddle(7, 64).conj(): re=cos(7π/32), im=sin(7π/32)
    twiddle3: Complex<T>,
    dct2_b8: Dct2Butterfly8<T>,
    dct2_b4: Dct2Butterfly4<T>,
}

impl<T: DctSample> Default for Dst2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 64).conj(),
            twiddle1: compute_twiddle(3, 64).conj(),
            twiddle2: compute_twiddle(5, 64).conj(),
            twiddle3: compute_twiddle(7, 64).conj(),
            dct2_b8: Dct2Butterfly8::default(),
            dct2_b4: Dct2Butterfly4::default(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let twiddles = [self.twiddle0, self.twiddle1, self.twiddle2, self.twiddle3];

        let mut dct2_buf = [T::default(); 8];
        let mut dct4_even = [T::default(); 4];
        let mut dct4_odd = [T::default(); 4];

        for i in 0usize..4 {
            let tw = twiddles[i];
            let (ib, it, ihb, iht) = if i % 2 == 0 {
                (data[i], -data[15 - i], -data[7 - i], data[8 + i])
            } else {
                (-data[i], data[15 - i], data[7 - i], -data[8 + i])
            };

            dct2_buf[i] = ib + it;
            dct2_buf[7 - i] = ihb + iht;

            let lower = ib - it;
            let upper = ihb - iht;

            dct4_even[i] = fmla(lower, tw.re, upper * tw.im);

            let sin_inp = fmla(upper, tw.re, -lower * tw.im);
            // Stored reversed with alternating sign: odd[3−i] = (−1)^i · sin_inp
            dct4_odd[3 - i] = if i % 2 == 0 { sin_inp } else { -sin_inp };
        }

        self.dct2_b8.exec(&mut InPlaceStore::new(&mut dct2_buf));
        self.dct2_b4.exec(&mut InPlaceStore::new(&mut dct4_even));
        self.dct2_b4.exec(&mut InPlaceStore::new(&mut dct4_odd));

        data[0] = -dct4_odd[0];
        data[1] = dct2_buf[7];
        data[2] = dct4_even[3] - dct4_odd[1];
        data[3] = dct2_buf[6];
        data[4] = dct4_even[3] + dct4_odd[1];
        data[5] = dct2_buf[5];
        data[6] = dct4_even[2] + dct4_odd[2];
        data[7] = dct2_buf[4];
        data[8] = dct4_even[2] - dct4_odd[2];
        data[9] = dct2_buf[3];
        data[10] = dct4_even[1] - dct4_odd[3];
        data[11] = dct2_buf[2];
        data[12] = dct4_even[1] + dct4_odd[3];
        data[13] = dct2_buf[1];
        data[14] = dct4_even[0];
        data[15] = dct2_buf[0];
    }
}

define_in_place_butterfly!(Dst2Butterfly16, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf_dst2, f64, Dst2Butterfly3, 3, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf5_dst2, f64, Dst2Butterfly5, 5, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf6_dst2, f64, Dst2Butterfly6, 6, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf7_dst2, f64, Dst2Butterfly7, 7, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf8_dst2, f64, Dst2Butterfly8, 8, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf9_dst2, f64, Dst2Butterfly9, 9, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf16_dst2, f64, Dst2Butterfly16, 16, 1e-7, naive_dst2);
}
