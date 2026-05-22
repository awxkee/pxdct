/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

/// Normalization mode applied on top of an executor's un-normalized output.
///
/// All executors in this crate produce results consistent with the textbook
/// (un-normalized) definition of each transform by default. `Scaling`
/// describes the post-processing the executor should apply automatically:
///
/// * [`Scaling::None`]  - no extra scaling; output is exactly the
///   un-normalized textbook form.
/// * [`Scaling::Scale`] - flat `sqrt(2 / N)` factor applied uniformly to every
///   output element, regardless of transform type. This is the same factor
///   used by the historical `make_scaled_*` family and is cheap and
///   predictable, but is *not* an orthonormal transform for most types.
/// * [`Scaling::Ortho`] - per-type orthonormal scaling so the transform
///   matrix is unitary. Forward/inverse pairs at the same length round-trip
///   to the identity up to floating-point error. Conventions follow SciPy /
///   Britanak for DCT-I..IV and the natural half-sample extension for the
///   "odd" types V..VIII.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Scaling {
    /// No post-processing.
    #[default]
    None,
    /// Multiply every output element by `sqrt(2 / N)`.
    Scale,
    /// Apply per-type orthonormal scaling.
    Ortho,
}

/// Identifier of a transform family. Used to dispatch the
/// per-type orthonormal scaling rule.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TransformKind {
    Dct1,
    Dst1,
    Dct2,
    Dct3,
    Dct4,
    Dst2,
    Dst3,
    Dst4,
    Dct5,
    Dst5,
    Dct6,
    Dst6,
    Dct7,
    Dst7,
    Dct8,
    Dst8,
}

/// How an input "correction" term (if any) is added to every output
/// element before flat scaling.
#[derive(Copy, Clone, Debug)]
enum InputCorrection<T> {
    /// No correction.
    None,
    /// Add `factor * input[index]` to every `output[k]`.
    Constant { index: usize, factor: T },
    /// Add `factor * input[index] * (-1)^k` to every `output[k]`.
    Alternating { index: usize, factor: T },
    /// DCT-I orthonormal correction: add
    /// `factor * input[0] + factor * input[N-1] * (-1)^k`
    /// to every `output[k]`. `factor` is pre-computed as `sqrt(2) - 1`.
    Dct1Dual { factor: T },
}

/// Recipe for post-processing an executor's output to realise a
/// [`Scaling`] mode for a particular [`TransformKind`].
#[derive(Copy, Clone, Debug)]
struct ScalingPlan<T> {
    /// Add this correction term to every output element first.
    correction: InputCorrection<T>,
    /// Multiply every output element by `flat`.
    flat: T,
    /// Whether `flat` is meaningfully different from `1`. (Set at construction
    /// time so we never have to test `PartialEq` on `T` at runtime.)
    apply_flat: bool,
    /// If `Some`, also multiply `y[0]` by this after the flat scale.
    endpoint_head: Option<T>,
    /// If `Some`, also multiply `y[N-1]` by this after the flat scale.
    endpoint_tail: Option<T>,
}

impl<T: DctSample> ScalingPlan<T>
where
    f64: AsPrimitive<T>,
{
    /// Flat `sqrt(2/N)` plan with no endpoint or correction adjustments.
    fn flat_only(length: usize) -> Self {
        let flat = (2.0_f64 / length as f64).sqrt().as_();
        Self {
            correction: InputCorrection::None,
            flat,
            apply_flat: true,
            endpoint_head: None,
            endpoint_tail: None,
        }
    }

    /// Build an Ortho plan for the given transform kind and length.
    ///
    /// The conventions chosen here are:
    ///
    /// * DCT-I..IV: SciPy `norm="ortho"` semantics (so we match
    ///   `scipy.fft.dct(..., type=n, norm='ortho')`).
    /// * DST-I..IV: SciPy `norm="ortho"` semantics for `scipy.fft.dst`.
    /// * DCT/DST V..VIII: the natural orthonormal extension following
    ///   Britanak/Yip/Rao. Each type's flat factor is `sqrt(2/(N ± 1/2))`,
    ///   plus an "endpoint" half-scaling on whichever output index has the
    ///   degenerate (half-amplitude) basis vector. Forward/inverse pairs
    ///   (V↔V, VI↔VII, VIII↔VIII) round-trip to the identity.
    fn ortho(kind: TransformKind, length: usize) -> Result<Self, PxdctError> {
        if length == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        let n = length as f64;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        // Correction for "0.5 implicit input weight" -> "1/sqrt(2)":
        //     naive contributes  0.5     * x[idx] * basis(k, idx)
        //     ortho needs       1/sqrt2 * x[idx] * basis(k, idx)
        let lift_half = inv_sqrt2 - 0.5_f64;

        let plan = match kind {
            // ---------------- Type I ----------------
            TransformKind::Dct1 => {
                // SciPy DCT-I ortho: x[0] and x[N-1] *= sqrt(2),
                // y[0] and y[N-1] /= sqrt(2), overall scale sqrt(1/(2(N-1))).
                //
                // We express the input pre-multiplication as an additive
                // correction. The naive form contributes 1*x[0] and
                // (-1)^k * x[N-1] to y[k]. After lifting both by sqrt(2),
                // those contributions become sqrt(2)*x[0] and sqrt(2)*(-1)^k*x[N-1],
                // so we add (sqrt(2)-1)*x[0] + (sqrt(2)-1)*(-1)^k*x[N-1] to each
                // y[k]. Then flat-scale by sqrt(1/(2(N-1))) and halve y[0], y[N-1].
                if length < 2 {
                    return Err(PxdctError::MinimumPoints(2, "DCT-I".to_string()));
                }
                let flat = (1.0_f64 / (2.0 * (n - 1.0))).sqrt();
                let dual_factor: T = (std::f64::consts::SQRT_2 - 1.0_f64).as_();
                Self {
                    correction: InputCorrection::Dct1Dual {
                        factor: dual_factor,
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: Some(inv_sqrt2.as_()),
                    endpoint_tail: Some(inv_sqrt2.as_()),
                }
            }
            TransformKind::Dst1 => {
                // SciPy DST-I ortho: flat sqrt(2/(N+1)), no endpoint or correction.
                let flat = (2.0_f64 / (n + 1.0)).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- Type II / III ----------------
            TransformKind::Dct2 => {
                let flat = (2.0_f64 / n).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: Some(inv_sqrt2.as_()),
                    endpoint_tail: None,
                }
            }
            TransformKind::Dct3 => {
                let flat = (2.0_f64 / n).sqrt();
                Self {
                    correction: InputCorrection::Constant {
                        index: 0,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- Type IV ----------------
            TransformKind::Dct4 | TransformKind::Dst4 => {
                let flat = (2.0_f64 / n).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- DST II / III ----------------
            TransformKind::Dst2 => {
                let flat = (2.0_f64 / n).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: Some(inv_sqrt2.as_()),
                }
            }
            TransformKind::Dst3 => {
                let flat = (2.0_f64 / n).sqrt();
                Self {
                    correction: InputCorrection::Alternating {
                        index: length - 1,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- DCT V / DST V (denominator N - 1/2 or N + 1/2) ----------------
            TransformKind::Dct5 => {
                // Naive has 0.5 multiplier at n=0; basis(k,0)=1 (constant correction).
                let denom = n - 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::Constant {
                        index: 0,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: Some(inv_sqrt2.as_()),
                    endpoint_tail: None,
                }
            }
            TransformKind::Dst5 => {
                let denom = n + 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- DCT VI / DST VI ----------------
            TransformKind::Dct6 => {
                // Naive has 0.5 multiplier at n=N-1; basis alternates (-1)^k.
                let denom = n - 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::Alternating {
                        index: length - 1,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: Some(inv_sqrt2.as_()),
                    endpoint_tail: None,
                }
            }
            TransformKind::Dst6 => {
                let denom = n + 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- DCT VII / DST VII ----------------
            TransformKind::Dct7 => {
                // Naive has 0.5 multiplier at n=0 (constant correction).
                let denom = n - 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::Constant {
                        index: 0,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: Some(inv_sqrt2.as_()),
                }
            }
            TransformKind::Dst7 => {
                let denom = n + 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            // ---------------- DCT VIII / DST VIII ----------------
            TransformKind::Dct8 => {
                let denom = n + 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::None,
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: None,
                }
            }
            TransformKind::Dst8 => {
                // Naive has 0.5 multiplier at n=N-1; basis alternates (-1)^k.
                let denom = n - 0.5;
                let flat = (2.0_f64 / denom).sqrt();
                Self {
                    correction: InputCorrection::Alternating {
                        index: length - 1,
                        factor: lift_half.as_(),
                    },
                    flat: flat.as_(),
                    apply_flat: true,
                    endpoint_head: None,
                    endpoint_tail: Some(inv_sqrt2.as_()),
                }
            }
        };

        Ok(plan)
    }
}

/// Generic post-processing wrapper that applies a [`ScalingPlan`] to an
/// inner executor's output.
pub(crate) struct ScalingInterceptor<T> {
    pub(crate) interceptor: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    plan: ScalingPlan<T>,
}

impl<T: DctSample> ScalingInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    /// Apply the post-processing step in-place to `output`, using the
    /// pre-saved input snapshot values needed by any active corrections.
    #[inline]
    fn post_process(&self, output: &mut [T], snapshot: PreSnapshot<T>) {
        match self.plan.correction {
            InputCorrection::None => {}
            InputCorrection::Constant { factor, .. } => {
                let add = factor * snapshot.correction_value;
                for y in output.iter_mut() {
                    *y += add;
                }
            }
            InputCorrection::Alternating { factor, .. } => {
                let add = factor * snapshot.correction_value;
                let minus_one: T = (-1.0_f64).as_();
                let neg_add = minus_one * add;
                for (k, y) in output.iter_mut().enumerate() {
                    if k & 1 == 0 {
                        *y += add;
                    } else {
                        *y += neg_add;
                    }
                }
            }
            InputCorrection::Dct1Dual { factor } => {
                let head_contrib = factor * snapshot.head_value;
                let tail_contrib = factor * snapshot.tail_value;
                let minus_one: T = (-1.0_f64).as_();
                let neg_tail = minus_one * tail_contrib;
                for (k, y) in output.iter_mut().enumerate() {
                    let alt_tail = if k & 1 == 0 { tail_contrib } else { neg_tail };
                    *y += head_contrib + alt_tail;
                }
            }
        }

        if self.plan.apply_flat {
            let flat = self.plan.flat;
            for y in output.iter_mut() {
                *y *= flat;
            }
        }

        if let Some(ep) = self.plan.endpoint_head
            && let Some(first) = output.first_mut()
        {
            *first *= ep;
        }
        if let Some(ep) = self.plan.endpoint_tail
            && let Some(last) = output.last_mut()
        {
            *last *= ep;
        }
    }

    /// Capture the input values the post-processing step will need,
    /// *before* the inner executor runs (so in-place execution can still
    /// access the original input values).
    #[inline]
    fn snapshot(&self, input: &[T]) -> PreSnapshot<T> {
        let zero: T = T::default();
        match self.plan.correction {
            InputCorrection::None => PreSnapshot {
                head_value: zero,
                tail_value: zero,
                correction_value: zero,
            },
            InputCorrection::Constant { index, .. }
            | InputCorrection::Alternating { index, .. } => PreSnapshot {
                head_value: zero,
                tail_value: zero,
                correction_value: input.get(index).copied().unwrap_or(zero),
            },
            InputCorrection::Dct1Dual { .. } => {
                let n = input.len();
                PreSnapshot {
                    head_value: input.first().copied().unwrap_or(zero),
                    tail_value: if n > 0 { input[n - 1] } else { zero },
                    correction_value: zero,
                }
            }
        }
    }
}

/// Tiny POD used to ferry pre-execution input values across the inner
/// executor call to the post-processing step.
#[derive(Copy, Clone)]
struct PreSnapshot<T> {
    head_value: T,
    tail_value: T,
    correction_value: T,
}

impl<T: DctSample> PxdctExecutor<T> for ScalingInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        let snap = self.snapshot(data);
        self.interceptor.execute_with_scratch(data, scratch)?;
        self.post_process(data, snap);
        Ok(())
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
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, self.interceptor.length());
        let snap = self.snapshot(input);
        self.interceptor
            .execute_into_with_scratch(input, output, scratch)?;
        self.post_process(output, snap);
        Ok(())
    }

    fn length(&self) -> usize {
        self.interceptor.length()
    }

    fn scratch_size(&self) -> usize {
        self.interceptor.scratch_size()
    }
}

/// Wrap an executor in a [`ScalingInterceptor`] so it applies the given
/// [`Scaling`] mode for its [`TransformKind`]. Returns the original
/// executor untouched when `Scaling::None` is requested.
pub(crate) fn wrap_with_scaling<T: DctSample>(
    inner: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    kind: TransformKind,
    scaling: Scaling,
) -> Result<Arc<dyn PxdctExecutor<T> + Send + Sync>, PxdctError>
where
    f64: AsPrimitive<T>,
{
    match scaling {
        Scaling::None => Ok(inner),
        Scaling::Scale => {
            let plan: ScalingPlan<T> = ScalingPlan::flat_only(inner.length());
            Ok(Arc::new(ScalingInterceptor {
                interceptor: inner,
                plan,
            }))
        }
        Scaling::Ortho => {
            let length = inner.length();
            let plan: ScalingPlan<T> = ScalingPlan::ortho(kind, length)?;
            Ok(Arc::new(ScalingInterceptor {
                interceptor: inner,
                plan,
            }))
        }
    }
}
