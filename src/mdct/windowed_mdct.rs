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

/// Windowed MDCT/IMDCT with 50% overlap-add.
use crate::mdct::{ImdctFft, MdctFft};
use crate::twiddles::FftTrigonometry;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use pxfm::f_i0;
use std::sync::Arc;

pub enum MdctChoiceWindow<T> {
    BuiltIn(MdctWindow),
    External(Vec<T>),
}

/// Choice of analysis/synthesis window for [`WindowedMdct`] / [`WindowedImdct`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MdctWindow {
    /// Sine window: `w[n] = sin(pi * (n + 0.5) / (2N))`.
    Sine,
    /// Vorbis window: `w[n] = sin(pi/2 * sin(pi * (n + 0.5) / (2N))^2)`.
    Vorbis,
    /// Kaiser-Bessel Derived with alpha.
    Kbd { alpha: f64 },
}

impl MdctWindow {
    /// Generate a `2 * n`-sample window vector for this window kind.
    pub(crate) fn generate<T: DctSample>(self, n: usize) -> Result<Vec<T>, PxdctError>
    where
        f64: AsPrimitive<T>,
    {
        if n < 2 {
            return Err(PxdctError::InvalidSizeMultiplier(n, 2));
        }
        let len = 2 * n;
        let mut w = try_vec![T::default(); len];

        match self {
            MdctWindow::Sine => {
                for (i, slot) in w.iter_mut().enumerate() {
                    let theta = (i as f64 + 0.5) / (2.0 * n as f64);
                    *slot = theta.sinpi().as_();
                }
            }
            MdctWindow::Vorbis => {
                for (i, slot) in w.iter_mut().enumerate() {
                    let inner = (i as f64 + 0.5) / (2.0 * n as f64);
                    let s = inner.sinpi();
                    let outer = 0.5 * s * s;
                    *slot = outer.sinpi().as_();
                }
            }
            MdctWindow::Kbd { alpha } => {
                let kaiser = kaiser_window_f64(n + 1, alpha * std::f64::consts::PI);
                let mut cum = try_vec![0.0_f64; n + 1];
                let mut acc = 0.0_f64;
                for (c, k) in cum.iter_mut().zip(kaiser.iter()) {
                    acc += *k;
                    *c = acc;
                }
                let total = *cum.last().unwrap();
                for i in 0..n {
                    let v = (cum[i] / total).sqrt();
                    w[i] = v.as_();
                    w[len - 1 - i] = v.as_();
                }
            }
        }
        Ok(w)
    }
}

/// Kaiser window of given length and shape parameter `beta = alpha * pi`.
fn kaiser_window_f64(len: usize, beta: f64) -> Vec<f64> {
    let denom = f_i0(beta);
    let half = (len as f64 - 1.0) / 2.0;
    (0..len)
        .map(|i| {
            let r = (i as f64 - half) / half;
            let arg = beta * (1.0 - r * r).max(0.0).sqrt();
            f_i0(arg) / denom
        })
        .collect()
}

/// Single-block windowed forward MDCT
pub struct WindowedMdct<T> {
    mdct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    window: Vec<T>,
    /// `n` MDCT bins; the block size is `2 * n`.
    n: usize,
    /// Scratch needed by the inner MDCT, in units of `T`.
    inner_scratch_size: usize,
}

impl<T: DctSample> WindowedMdct<T>
where
    f64: AsPrimitive<T>,
{
    /// Build a windowed forward MDCT for `n` output bins (input block = `2n`).
    pub(crate) fn new(n: usize, window: MdctWindow) -> Result<Self, PxdctError> {
        let inner = MdctFft::<T>::new(n)?;
        let inner_scratch_size = inner.scratch_size();
        let mdct: Arc<dyn PxdctExecutor<T> + Send + Sync> = Arc::new(inner);
        let w = window.generate::<T>(n)?;
        Ok(Self {
            mdct,
            window: w,
            n,
            inner_scratch_size,
        })
    }

    /// Build with a caller-supplied window vector of length `2 * n`.
    pub(crate) fn with_window(n: usize, window: Vec<T>) -> Result<Self, PxdctError> {
        if window.len() != 2 * n {
            return Err(PxdctError::InvalidSizeMultiplier(window.len(), 2 * n));
        }
        let inner = MdctFft::<T>::new(n)?;
        let inner_scratch_size = inner.scratch_size();
        let mdct: Arc<dyn PxdctExecutor<T> + Send + Sync> = Arc::new(inner);
        Ok(Self {
            mdct,
            window,
            n,
            inner_scratch_size,
        })
    }

    pub(crate) fn window(&self) -> &[T] {
        &self.window
    }
}

impl<T: DctSample> PxdctExecutor<T> for WindowedMdct<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _data: &mut [T]) -> Result<(), PxdctError> {
        // Windowed forward MDCT cannot be in-place: input length is 2N,
        // output is N.
        Err(PxdctError::InvalidSizeMultiplier(2 * self.n, self.n))
    }

    fn execute_with_scratch(&self, _data: &mut [T], _scratch: &mut [T]) -> Result<(), PxdctError> {
        Err(PxdctError::InvalidSizeMultiplier(2 * self.n, self.n))
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let in_block = 2 * self.n;
        let out_block = self.n;
        if !input.len().is_multiple_of(in_block) {
            return Err(PxdctError::InvalidSizeMultiplier(input.len(), in_block));
        }
        if !output.len().is_multiple_of(out_block) {
            return Err(PxdctError::InvalidSizeMultiplier(output.len(), out_block));
        }
        if input.len() / in_block != output.len() / out_block {
            return Err(PxdctError::InvalidSizeMultiplier(
                output.len(),
                input.len() / 2,
            ));
        }

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (windowed, inner_scratch) = full_scratch.split_at_mut(in_block);

        for (src, dst) in input
            .chunks_exact(in_block)
            .zip(output.chunks_exact_mut(out_block))
        {
            for (w_slot, (s, w)) in windowed.iter_mut().zip(src.iter().zip(self.window.iter())) {
                *w_slot = *s * *w;
            }
            self.mdct
                .execute_into_with_scratch(windowed, dst, inner_scratch)?;
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.n
    }

    fn scratch_size(&self) -> usize {
        // A windowed work-block (2N) + whatever the inner MDCT needs.
        2 * self.n + self.inner_scratch_size
    }
}

/// Single-block windowed inverse MDCT: runs the raw IMDCT, then applies a
/// window to the 2N-sample output.
///
/// Implements [`PxdctExecutor`] like [`WindowedMdct`]. Leaves overlap-add to
/// the caller; use [`ImdctOverlapAdd`] for the full streaming pipeline.
pub struct WindowedImdct<T> {
    imdct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    window: Vec<T>,
    length: usize,
    inner_scratch_size: usize,
}

impl<T: DctSample> WindowedImdct<T>
where
    f64: AsPrimitive<T>,
{
    /// Build a windowed inverse MDCT for `n` input bins (output block = `2n`).
    pub(crate) fn new(n: usize, window: MdctWindow) -> Result<Self, PxdctError> {
        let inner = ImdctFft::<T>::new(n)?;
        let inner_scratch_size = inner.scratch_size();
        let imdct: Arc<dyn PxdctExecutor<T> + Send + Sync> = Arc::new(inner);
        let w = window.generate::<T>(n)?;
        Ok(Self {
            imdct,
            window: w,
            length: n,
            inner_scratch_size,
        })
    }

    /// Build with a caller-supplied window vector of length `2 * n`.
    pub(crate) fn with_window(n: usize, window: Vec<T>) -> Result<Self, PxdctError> {
        if window.len() != 2 * n {
            return Err(PxdctError::InvalidSizeMultiplier(window.len(), 2 * n));
        }
        let inner = ImdctFft::<T>::new(n)?;
        let inner_scratch_size = inner.scratch_size();
        let imdct: Arc<dyn PxdctExecutor<T> + Send + Sync> = Arc::new(inner);
        Ok(Self {
            imdct,
            window,
            length: n,
            inner_scratch_size,
        })
    }

    /// Read-only access to the synthesis window (length `2 * n`).
    pub(crate) fn window(&self) -> &[T] {
        &self.window
    }
}

impl<T: DctSample> PxdctExecutor<T> for WindowedImdct<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _data: &mut [T]) -> Result<(), PxdctError> {
        Err(PxdctError::InvalidSizeMultiplier(
            self.length,
            2 * self.length,
        ))
    }

    fn execute_with_scratch(&self, _data: &mut [T], _scratch: &mut [T]) -> Result<(), PxdctError> {
        Err(PxdctError::InvalidSizeMultiplier(
            self.length,
            2 * self.length,
        ))
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let in_block = self.length;
        let out_block = 2 * self.length;
        if !input.len().is_multiple_of(in_block) {
            return Err(PxdctError::InvalidSizeMultiplier(input.len(), in_block));
        }
        if !output.len().is_multiple_of(out_block) {
            return Err(PxdctError::InvalidSizeMultiplier(output.len(), out_block));
        }
        if input.len() / in_block != output.len() / out_block {
            return Err(PxdctError::InvalidSizeMultiplier(
                output.len(),
                input.len() * 2,
            ));
        }

        let inner_scratch = validate_scratch!(scratch, self.scratch_size());

        for (src, dst) in input
            .chunks_exact(in_block)
            .zip(output.chunks_exact_mut(out_block))
        {
            self.imdct
                .execute_into_with_scratch(src, dst, inner_scratch)?;
            for (slot, w) in dst.iter_mut().zip(self.window.iter()) {
                *slot *= *w;
            }
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.length
    }

    fn scratch_size(&self) -> usize {
        // The IMDCT writes directly into the output block, so we only need
        // whatever scratch the inner IMDCT itself wants.
        self.inner_scratch_size
    }
}

/// Streaming forward MDCT with internal overlap buffer.
pub struct MdctOverlapAdd<T> {
    mdct: WindowedMdct<T>,
    /// Holds the previous hop (`n` samples) so the next call can assemble a
    /// `2n`-sample windowed block.
    prev: Vec<T>,
    /// Reusable `2n`-sample assembly buffer for the prev||hop block.
    block: Vec<T>,
    /// Reusable scratch for the inner windowed MDCT.
    scratch: Vec<T>,
}

pub trait TransformOverlapAdd<T> {
    /// Process one `n`-sample hop and produce `n` MDCT coefficients.
    fn execute(&mut self, hop: &[T], coeffs: &mut [T]) -> Result<(), PxdctError>;
    /// Process `k` contiguous hops in one call: `hops` is `k * n` audio
    /// samples, `coeffs` is `k * n` MDCT bins.
    fn execute_many(&mut self, hops: &[T], coeffs: &mut [T]) -> Result<(), PxdctError>;
    /// Reset the overlap buffer to zeros. Call between independent streams.
    fn reset(&mut self);
    /// Reset the overlap buffer to zeros. Call between independent streams.
    fn length(&self) -> usize;
    /// Returns current window
    fn window(&self) -> &[T];
}

impl<T: DctSample> MdctOverlapAdd<T>
where
    f64: AsPrimitive<T>,
{
    /// Build a streaming windowed MDCT for hop size `n`.
    pub(crate) fn new(n: usize, window: MdctWindow) -> Result<Self, PxdctError> {
        let mdct = WindowedMdct::new(n, window)?;
        let scratch = try_vec![T::default(); mdct.scratch_size()];
        Ok(Self {
            prev: try_vec![T::default(); n],
            block: try_vec![T::default(); 2 * n],
            scratch,
            mdct,
        })
    }

    /// Build a streaming windowed MDCT for hop size `n`.
    pub(crate) fn new_with_window(n: usize, window: Vec<T>) -> Result<Self, PxdctError> {
        let mdct = WindowedMdct::with_window(n, window)?;
        let scratch = try_vec![T::default(); mdct.scratch_size()];
        Ok(Self {
            prev: try_vec![T::default(); n],
            block: try_vec![T::default(); 2 * n],
            scratch,
            mdct,
        })
    }
}

impl<T: DctSample> TransformOverlapAdd<T> for MdctOverlapAdd<T>
where
    f64: AsPrimitive<T>,
{
    /// Process one `n`-sample hop and produce `n` MDCT coefficients.
    fn execute(&mut self, hop: &[T], coeffs: &mut [T]) -> Result<(), PxdctError> {
        let n = self.mdct.n;
        if hop.len() != n {
            return Err(PxdctError::InvalidSizeMultiplier(hop.len(), n));
        }
        if coeffs.len() != n {
            return Err(PxdctError::InvalidSizeMultiplier(coeffs.len(), n));
        }

        // Assemble the 2N-sample block: previous hop + current hop.
        self.block[..n].copy_from_slice(&self.prev);
        self.block[n..].copy_from_slice(hop);

        self.mdct
            .execute_into_with_scratch(&self.block, coeffs, &mut self.scratch)?;

        // Save current hop as the prev for the next call.
        self.prev.copy_from_slice(hop);
        Ok(())
    }

    /// Process `k` contiguous hops in one call: `hops` is `k * n` audio
    /// samples, `coeffs` is `k * n` MDCT bins.
    fn execute_many(&mut self, hops: &[T], coeffs: &mut [T]) -> Result<(), PxdctError> {
        let n = self.mdct.n;
        if !hops.len().is_multiple_of(n) {
            return Err(PxdctError::InvalidSizeMultiplier(hops.len(), n));
        }
        if !coeffs.len().is_multiple_of(n) {
            return Err(PxdctError::InvalidSizeMultiplier(coeffs.len(), n));
        }
        if hops.len() != coeffs.len() {
            return Err(PxdctError::InvalidSizeMultiplier(coeffs.len(), hops.len()));
        }
        for (hop, c) in hops.chunks_exact(n).zip(coeffs.chunks_exact_mut(n)) {
            self.execute(hop, c)?;
        }
        Ok(())
    }

    /// Reset the overlap buffer to zeros. Call between independent streams.
    fn reset(&mut self) {
        for slot in self.prev.iter_mut() {
            *slot = T::default();
        }
    }

    /// Hop size (== `n` MDCT bins).
    fn length(&self) -> usize {
        self.mdct.n
    }

    fn window(&self) -> &[T] {
        self.mdct.window()
    }
}

/// Streaming inverse MDCT with internal overlap-add buffer.
pub struct ImdctOverlapAdd<T> {
    imdct: WindowedImdct<T>,
    /// Tail half of the previous block, to be added to the next call's head.
    tail: Vec<T>,
    /// Reusable `2n`-sample synthesis buffer.
    block: Vec<T>,
    /// Reusable scratch for the inner windowed IMDCT.
    scratch: Vec<T>,
}

impl<T: DctSample> ImdctOverlapAdd<T>
where
    f64: AsPrimitive<T>,
{
    /// Build a streaming windowed IMDCT for hop size `n`.
    pub(crate) fn new(n: usize, window: MdctWindow) -> Result<Self, PxdctError> {
        let imdct = WindowedImdct::new(n, window)?;
        let scratch = try_vec![T::default(); imdct.scratch_size()];
        Ok(Self {
            tail: try_vec![T::default(); n],
            block: try_vec![T::default(); 2 * n],
            scratch,
            imdct,
        })
    }

    /// Build a streaming windowed IMDCT for hop size `n`.
    pub(crate) fn new_with_window(n: usize, window: Vec<T>) -> Result<Self, PxdctError> {
        let imdct = WindowedImdct::with_window(n, window)?;
        let scratch = try_vec![T::default(); imdct.scratch_size()];
        Ok(Self {
            tail: try_vec![T::default(); n],
            block: try_vec![T::default(); 2 * n],
            scratch,
            imdct,
        })
    }
}

impl<T: DctSample> TransformOverlapAdd<T> for ImdctOverlapAdd<T>
where
    f64: AsPrimitive<T>,
{
    /// Process one block of `n` MDCT coefficients and produce `n` audio samples.
    fn execute(&mut self, coeffs: &[T], out: &mut [T]) -> Result<(), PxdctError> {
        let n = self.imdct.length;
        if coeffs.len() != n {
            return Err(PxdctError::InvalidSizeMultiplier(coeffs.len(), n));
        }
        if out.len() != n {
            return Err(PxdctError::InvalidSizeMultiplier(out.len(), n));
        }

        self.imdct
            .execute_into_with_scratch(coeffs, &mut self.block, &mut self.scratch)?;

        // Overlap-add: out[i] = block[i] + tail[i]; new tail = block[n..2n].
        let (head, new_tail) = self.block.split_at(n);
        for (dst, (h, t)) in out.iter_mut().zip(head.iter().zip(self.tail.iter())) {
            *dst = *h + *t;
        }
        self.tail.copy_from_slice(new_tail);
        Ok(())
    }

    /// Process `k` contiguous blocks in one call: `coeffs` is `k * n` MDCT
    /// bins, `out` is `k * n` audio samples.
    fn execute_many(&mut self, coeffs: &[T], out: &mut [T]) -> Result<(), PxdctError> {
        let n = self.imdct.length;
        if !coeffs.len().is_multiple_of(n) {
            return Err(PxdctError::InvalidSizeMultiplier(coeffs.len(), n));
        }
        if !out.len().is_multiple_of(n) {
            return Err(PxdctError::InvalidSizeMultiplier(out.len(), n));
        }
        if coeffs.len() != out.len() {
            return Err(PxdctError::InvalidSizeMultiplier(out.len(), coeffs.len()));
        }
        for (c, o) in coeffs.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            self.execute(c, o)?;
        }
        Ok(())
    }

    /// Reset the overlap-add buffer to zeros.
    fn reset(&mut self) {
        for slot in self.tail.iter_mut() {
            *slot = T::default();
        }
    }

    fn length(&self) -> usize {
        self.imdct.length
    }

    fn window(&self) -> &[T] {
        self.imdct.window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Princen-Bradley identity: w[n]^2 + w[n + N]^2 == 1 for all n in 0..N.
    fn check_princen_bradley(w: &[f64], n: usize, tol: f64) {
        assert_eq!(w.len(), 2 * n);
        for i in 0..n {
            let v = w[i] * w[i] + w[i + n] * w[i + n];
            assert!(
                (v - 1.0).abs() < tol,
                "Princen-Bradley violated at i={i}: {v}"
            );
        }
    }

    #[test]
    fn sine_window_satisfies_princen_bradley() {
        for n in [4, 8, 16, 64, 256] {
            let w = MdctWindow::Sine.generate::<f64>(n).unwrap();
            check_princen_bradley(&w, n, 1e-14);
        }
    }

    #[test]
    fn vorbis_window_satisfies_princen_bradley() {
        for n in [4, 8, 16, 64, 256] {
            let w = MdctWindow::Vorbis.generate::<f64>(n).unwrap();
            check_princen_bradley(&w, n, 1e-14);
        }
    }

    #[test]
    fn kbd_window_satisfies_princen_bradley() {
        for &alpha in &[4.0, 5.0, 6.0] {
            for n in [8, 16, 64, 256] {
                let w = MdctWindow::Kbd { alpha }.generate::<f64>(n).unwrap();
                check_princen_bradley(&w, n, 1e-12);
            }
        }
    }

    fn run_roundtrip(n: usize, window: MdctWindow, tol: f64) {
        let mut enc = MdctOverlapAdd::<f64>::new(n, window).unwrap();
        let mut dec = ImdctOverlapAdd::<f64>::new(n, window).unwrap();

        let num_hops = 6;
        let signal: Vec<f64> = (0..num_hops * n)
            .map(|i| ((i as f64) * 0.13).sin() + 0.4 * ((i as f64) * 0.07).cos())
            .collect();

        let mut coeffs = vec![0.0_f64; n];
        let mut out = vec![0.0_f64; n];
        let mut reconstructed = vec![0.0_f64; num_hops * n];

        for h in 0..num_hops {
            let hop_in = &signal[h * n..(h + 1) * n];
            enc.execute(hop_in, &mut coeffs).unwrap();
            dec.execute(&coeffs, &mut out).unwrap();
            reconstructed[h * n..(h + 1) * n].copy_from_slice(&out);
        }

        let scale = (n as f64) / 2.0;
        let start = 2 * n;
        let end = reconstructed.len();
        let src_start = n;
        let src_end = src_start + (end - start);

        for (i, (&r, &s)) in reconstructed[start..end]
            .iter()
            .zip(signal[src_start..src_end].iter())
            .enumerate()
        {
            let expected = scale * s;
            assert!(
                (r - expected).abs() < tol * (1.0 + expected.abs()),
                "roundtrip mismatch at sample {} (hop {}): got {}, expected {}",
                start + i,
                (start + i) / n,
                r,
                expected
            );
        }
    }

    #[test]
    fn roundtrip_sine_window() {
        run_roundtrip(8, MdctWindow::Sine, 1e-9);
        run_roundtrip(32, MdctWindow::Sine, 1e-9);
        run_roundtrip(128, MdctWindow::Sine, 1e-9);
    }

    #[test]
    fn roundtrip_vorbis_window() {
        run_roundtrip(8, MdctWindow::Vorbis, 1e-9);
        run_roundtrip(32, MdctWindow::Vorbis, 1e-9);
        run_roundtrip(128, MdctWindow::Vorbis, 1e-9);
    }

    #[test]
    fn roundtrip_kbd_window() {
        run_roundtrip(32, MdctWindow::Kbd { alpha: 4.0 }, 1e-9);
        run_roundtrip(128, MdctWindow::Kbd { alpha: 4.0 }, 1e-9);
        run_roundtrip(128, MdctWindow::Kbd { alpha: 6.0 }, 1e-9);
    }

    #[test]
    fn reset_clears_overlap_buffer() {
        let mut enc = MdctOverlapAdd::<f64>::new(16, MdctWindow::Sine).unwrap();
        let mut coeffs = vec![0.0_f64; 16];
        let hop: Vec<f64> = (1..=16).map(|i| i as f64).collect();

        enc.execute(&hop, &mut coeffs).unwrap();
        enc.reset();

        let zero_hop = vec![0.0_f64; 16];
        let mut after_reset = vec![0.0_f64; 16];
        enc.execute(&zero_hop, &mut after_reset).unwrap();

        let mut fresh = MdctOverlapAdd::<f64>::new(16, MdctWindow::Sine).unwrap();
        let mut fresh_out = vec![0.0_f64; 16];
        fresh.execute(&zero_hop, &mut fresh_out).unwrap();

        for (a, b) in after_reset.iter().zip(fresh_out.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn windowed_single_block_size_validation() {
        let mdct = WindowedMdct::<f64>::new(16, MdctWindow::Sine).unwrap();
        let mut out = vec![0.0_f64; 16];
        // Wrong input length:
        assert!(mdct.execute_into(&vec![0.0_f64; 8], &mut out).is_err());
        // Wrong output length:
        assert!(
            mdct.execute_into(&vec![0.0_f64; 32], &mut vec![0.0_f64; 8])
                .is_err()
        );
    }

    #[test]
    fn custom_window_rejects_wrong_length() {
        assert!(WindowedMdct::<f64>::with_window(16, vec![0.0; 31]).is_err());
        assert!(WindowedImdct::<f64>::with_window(16, vec![0.0; 31]).is_err());
    }

    #[test]
    fn windowed_mdct_batched_matches_single_block() {
        let n = 32;
        let mdct = WindowedMdct::<f64>::new(n, MdctWindow::Sine).unwrap();

        let block_a: Vec<f64> = (0..2 * n).map(|i| (i as f64 * 0.17).sin()).collect();
        let block_b: Vec<f64> = (0..2 * n).map(|i| (i as f64 * 0.09).cos()).collect();

        let mut single_a = vec![0.0_f64; n];
        let mut single_b = vec![0.0_f64; n];
        mdct.execute_into(&block_a, &mut single_a).unwrap();
        mdct.execute_into(&block_b, &mut single_b).unwrap();

        let mut combined_in = Vec::with_capacity(4 * n);
        combined_in.extend_from_slice(&block_a);
        combined_in.extend_from_slice(&block_b);
        let mut combined_out = vec![0.0_f64; 2 * n];
        mdct.execute_into(&combined_in, &mut combined_out).unwrap();

        for (i, (&c, &s)) in combined_out[..n].iter().zip(single_a.iter()).enumerate() {
            assert!(
                (c - s).abs() < 1e-12,
                "block A bin {i}: batched={c}, single={s}"
            );
        }
        for (i, (&c, &s)) in combined_out[n..].iter().zip(single_b.iter()).enumerate() {
            assert!(
                (c - s).abs() < 1e-12,
                "block B bin {i}: batched={c}, single={s}"
            );
        }
    }

    #[test]
    fn windowed_imdct_batched_matches_single_block() {
        let n = 32;
        let imdct = WindowedImdct::<f64>::new(n, MdctWindow::Sine).unwrap();

        let coeffs_a: Vec<f64> = (1..=n).map(|i| i as f64 * 0.3).collect();
        let coeffs_b: Vec<f64> = (1..=n).map(|i| (n - i) as f64 * 0.2).collect();

        let mut single_a = vec![0.0_f64; 2 * n];
        let mut single_b = vec![0.0_f64; 2 * n];
        imdct.execute_into(&coeffs_a, &mut single_a).unwrap();
        imdct.execute_into(&coeffs_b, &mut single_b).unwrap();

        let mut combined_in = Vec::with_capacity(2 * n);
        combined_in.extend_from_slice(&coeffs_a);
        combined_in.extend_from_slice(&coeffs_b);
        let mut combined_out = vec![0.0_f64; 4 * n];
        imdct.execute_into(&combined_in, &mut combined_out).unwrap();

        for (i, (&c, &s)) in combined_out[..2 * n]
            .iter()
            .zip(single_a.iter())
            .enumerate()
        {
            assert!(
                (c - s).abs() < 1e-10,
                "block A sample {i}: batched={c}, single={s}"
            );
        }
        for (i, (&c, &s)) in combined_out[2 * n..]
            .iter()
            .zip(single_b.iter())
            .enumerate()
        {
            assert!(
                (c - s).abs() < 1e-10,
                "block B sample {i}: batched={c}, single={s}"
            );
        }
    }

    #[test]
    fn windowed_executor_in_place_rejected() {
        let mdct = WindowedMdct::<f64>::new(16, MdctWindow::Sine).unwrap();
        let mut buf = vec![0.0_f64; 32];
        assert!(mdct.execute(&mut buf).is_err());

        let imdct = WindowedImdct::<f64>::new(16, MdctWindow::Sine).unwrap();
        let mut buf = vec![0.0_f64; 16];
        assert!(imdct.execute(&mut buf).is_err());
    }

    #[test]
    fn windowed_executor_mismatched_block_count_rejected() {
        let mdct = WindowedMdct::<f64>::new(16, MdctWindow::Sine).unwrap();
        // 2 input blocks (2 * 32 = 64) but only 1 output block worth (16).
        let input = vec![0.0_f64; 64];
        let mut output = vec![0.0_f64; 16];
        assert!(mdct.execute_into(&input, &mut output).is_err());
    }

    #[test]
    fn process_many_matches_single_calls() {
        let n = 16;
        let num_hops = 4;
        let mut enc_batched = MdctOverlapAdd::<f64>::new(n, MdctWindow::Sine).unwrap();
        let mut enc_single = MdctOverlapAdd::<f64>::new(n, MdctWindow::Sine).unwrap();

        let signal: Vec<f64> = (0..num_hops * n)
            .map(|i| ((i as f64) * 0.21).sin())
            .collect();

        let mut batched = vec![0.0_f64; num_hops * n];
        enc_batched.execute_many(&signal, &mut batched).unwrap();

        let mut single = vec![0.0_f64; num_hops * n];
        for h in 0..num_hops {
            enc_single
                .execute(&signal[h * n..(h + 1) * n], &mut single[h * n..(h + 1) * n])
                .unwrap();
        }

        for (i, (&b, &s)) in batched.iter().zip(single.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-14,
                "mismatch at {i}: batched={b}, single={s}"
            );
        }
    }
}
