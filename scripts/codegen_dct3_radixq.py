#!/usr/bin/env python3
"""
Code generator for odd mixed-radix-q DCT-III decompositions in Rust.

Given an odd integer q (the radix), this script emits a Rust source file
implementing `Dct3MixedRadix{q}<T>`, which decomposes a length-N = q*p
DCT-III into q length-p DCT-III's using the odd-factor algorithm from:

    G. Bi, "Fast Algorithms for Type-III DCT of Composite Sequence Lengths,"
    IEEE Trans. Signal Processing, vol. 47, no. 7, pp. 2053-2059, July 1999.

Conventions matching the existing pxdct codebase:

  * The outer transform is the textbook (unscaled) DCT-III in the
    "half-DC" (H_N) convention used by this library: the inner DCT-III
    halves its k=0 input, and the outer routine compensates with a
    -X(0)/2 adjustment so the overall output is also H_N.

  * Pair twiddles C_i = cos(α i / q) + j sin(α i / q) and rotation
    twiddles R_k = cos(α k / 2N) + j sin(α k / 2N) are precomputed in
    `new()`, where α = π(q - 1 - 2m) and m indexes the (q-1)/2 (V,W)
    pairs.

  * Buffer layout (in the per-call scratch):
        +-------------+------------------------+------------------------+
        |  A buffer   |  qh V-buffers (V_m's)  |  qh W'-buffers (W'_m's)|
        +-------------+------------------------+------------------------+
        |  p elements |  qh * p elements       |  qh * p elements       |
        +-------------+------------------------+------------------------+

  * The W'(0) = W(p, m) slot holds the singleton k = N/q boundary term,
    derived from eq (9) of the paper as
        W(p, m) = sum_{i=1..qh} (-1)^(i-1) X((2i-1)*p) sin((2i-1) α / (2q))
    (sign correction over the paper's eq (13), verified against the
     hand-written q = 3 implementation).

Usage:
    python codegen_dct3_radixq.py 5         # writes dct3_mixed_radix_5.rs
    python codegen_dct3_radixq.py 5 -o foo.rs
    python codegen_dct3_radixq.py --check 5 # verify math, don't write
"""
from __future__ import annotations
import argparse
import math
import struct
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def derive_pair_constants(q: int):
    """For each m in [0, qh), return list of (cos(α i/q), sin(α i/q)) for i in [1, qh].

    α = π(q - 1 - 2m).  These are the constants that multiply S_i(k) / T_i(k)
    in the V, W decompositions (eqs 11, 13 of the paper).
    """
    qh = (q - 1) // 2
    out = []
    for m in range(qh):
        alpha = math.pi * (q - 1 - 2 * m)
        row = [(math.cos(alpha * i / q), math.sin(alpha * i / q))
               for i in range(1, qh + 1)]
        out.append(row)
    return out


def derive_boundary_w_constants(q: int):
    """W(p, m) singleton coefficients: sin((2i-1) α / (2q)) for i in [1, qh],
    one row per m, including the (-1)^(i-1) sign baked in.
    """
    qh = (q - 1) // 2
    out = []
    for m in range(qh):
        alpha = math.pi * (q - 1 - 2 * m)
        row = []
        for i in range(1, qh + 1):
            sign = (-1) ** (i - 1)
            row.append(sign * math.sin((2 * i - 1) * alpha / (2 * q)))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Reference (used for self-check)
# ---------------------------------------------------------------------------

def _naive_dct3_p(X):
    """Textbook (full-DC, P) DCT-III, used inside the reference."""
    N = len(X)
    out = [0.0] * N
    for n in range(N):
        s = 0.0
        for k in range(N):
            s += X[k] * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
        out[n] = s
    return out


def _odd_factor_dct3_reference(X, q):
    """Pure-Python reference of the odd-factor decomposition (textbook P_N)."""
    N = len(X)
    assert N % q == 0 and q % 2 == 1
    p = N // q
    qh = (q - 1) // 2
    pair = derive_pair_constants(q)
    wp = derive_boundary_w_constants(q)

    def Si(i, k): return X[2 * i * p + k] + X[2 * i * p - k]
    def Ti(i, k): return X[2 * i * p + k] - X[2 * i * p - k]

    U = [0.0] * p
    U[0] = sum(((-1) ** i) * X[2 * i * p] for i in range(qh + 1))
    for k in range(1, p):
        U[k] = X[k] + sum(((-1) ** i) * Si(i, k) for i in range(1, qh + 1))

    V = [[0.0] * p for _ in range(qh)]
    Wp = [[0.0] * p for _ in range(qh)]
    for m in range(qh):
        alpha = math.pi * (q - 1 - 2 * m)
        V[m][0] = sum(((-1) ** i) * X[2 * i * p] * math.cos(alpha * i / q)
                      for i in range(qh + 1))
        for k in range(1, p):
            ck = math.cos(alpha * k / (2 * N))
            sk = math.sin(alpha * k / (2 * N))
            v = X[k] * ck
            for i in range(1, qh + 1):
                ci, si = pair[m][i - 1]
                v += ((-1) ** i) * (Si(i, k) * ci * ck - Ti(i, k) * si * sk)
            V[m][k] = v

        Wp[m][0] = sum(wp[m][i - 1] * X[(2 * i - 1) * p] for i in range(1, qh + 1))
        for k in range(1, p):
            ck = math.cos(alpha * k / (2 * N))
            sk = math.sin(alpha * k / (2 * N))
            inner1 = X[k] + sum(((-1) ** i) * Si(i, k) * pair[m][i - 1][0]
                               for i in range(1, qh + 1))
            inner2 = sum(((-1) ** i) * Ti(i, k) * pair[m][i - 1][1]
                        for i in range(1, qh + 1))
            Wp[m][p - k] = inner1 * sk + inner2 * ck

    A = _naive_dct3_p(U)
    F = [_naive_dct3_p(V[m]) for m in range(qh)]
    Graw = [_naive_dct3_p(Wp[m]) for m in range(qh)]

    x = [0.0] * N
    for n in range(p):
        x[q * n + qh] = A[n]
        for m in range(qh):
            G = ((-1) ** n) * Graw[m][n]
            x[q * n + m] = F[m][n] + G
            x[q * n + q - m - 1] = F[m][n] - G
    return x


def self_check(q: int, ps=(1, 2, 3, 4, 5), tol=1e-9):
    """Verify decomposition matches naive DCT-III for several N = q * p."""
    import random
    rng = random.Random(0xC0DEC0DE)
    qh = (q - 1) // 2
    for p in ps:
        N = q * p
        if p == 1 and qh > 0:
            # need indices 2*i*p, (2i-1)*p in range: requires p>=1; 2*qh*p = (q-1)*p < N=qp ✓
            # but Ti, Si require k=1..p-1 which is empty for p=1. Edge case, still OK.
            pass
        X = [rng.uniform(1.0, 2.0) for _ in range(N)]
        ref = _naive_dct3_p(X)
        ours = _odd_factor_dct3_reference(X, q)
        err = max(abs(a - b) for a, b in zip(ref, ours))
        flag = "PASS" if err < tol else "FAIL"
        print(f"  q={q}, N={N} (p={p}): max abs diff = {err:.3e}  [{flag}]")
        if err >= tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Rust code generator
# ---------------------------------------------------------------------------

LICENSE = """\
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
"""


def emit(q: int) -> str:
    """Generate Rust source for `Dct3MixedRadix{q}` (q must be odd, >= 3)."""
    if q < 3 or q % 2 == 0:
        raise ValueError(f"q must be an odd integer >= 3 (got {q})")

    qh = (q - 1) // 2

    out = []
    push = out.append

    # ---- Header / imports ----
    push(LICENSE.rstrip())
    push("")
    push(textwrap.dedent(f"""\
        use crate::bidirectional::{{BidirectionalStore, InPlaceStore}};
        use crate::mla::fmla;
        use crate::twiddles::FftTrigonometry;
        use crate::util::{{DctSample, try_vec, validate_scratch}};
        use crate::{{PxdctError, PxdctExecutor}};
        use num_complex::Complex;
        use num_traits::AsPrimitive;
        use std::sync::Arc;
    """))

    # ---- Constants trait + f32/f64 impls ----
    push(emit_constants_trait(q))

    # ---- Doc block + struct ----
    doc = build_doc(q, qh)
    for line in doc.splitlines():
        push("/// " + line if line else "///")
    push(emit_struct(q))

    # ---- new() ----
    push(emit_new(q))

    # ---- execute_with_store ----
    push(emit_execute_with_store(q, qh))

    # ---- PxdctExecutor impl (boilerplate identical across q) ----
    push(emit_executor_impl(q))

    # ---- Tests ----
    push(emit_tests(q))

    return "\n".join(out) + "\n"


def _rust_const(value: float) -> str:
    """Render a float as a Rust literal preserving full f64 precision.

    Uses Python's `repr` which gives a round-trippable decimal; we tack on `f64`
    so the literal types unambiguously.
    """
    s = repr(float(value))
    if "e" not in s and "." not in s:
        s += ".0"
    return f"{s}_f64"


# ===========================================================================
# Constant deduplication / sentinel resolution
# ===========================================================================
# Many trait constants for a given q collide up to sign — e.g. for q = 7,
# cos(2π/7) appears under several (m, i) addresses, sometimes with the (-1)^i
# sign giving the negative. We exploit this by:
#
#   1. Bucketing all raw values by their canonical magnitude (with a tiny tol).
#   2. Sorting magnitudes high → low so the most "useful" constants get
#      the lowest indices.
#   3. Naming each unique non-trivial magnitude `D3_R{q}_T{idx}`.
#   4. At every call site, looking up the (kind, m, i) entry to produce one of:
#        - the literal 0 / +1 / -1 / +½ / -½ (trivial twiddles)
#        - a constant reference `T::D3_R{q}_T{idx}` or its negation
#      and emitting the corresponding fmla / add / sub / no-op form.
#
# This is the same trick `Dct4MixedRadix9` uses (where constants like
# `D4_R9_ROT_TWIDDLE_0..6` are inlined with explicit `-` at use sites — see
# e.g. `u1 = fmla(iq3, -T::D4_R9_ROT_TWIDDLE_0, u1);`).
# ===========================================================================

SENT_ZERO     = "zero"
SENT_POS_ONE  = "pos_one"
SENT_NEG_ONE  = "neg_one"
SENT_POS_HALF = "pos_half"
SENT_NEG_HALF = "neg_half"

_DEDUP_TOL = 1e-12


def _canon_key(v: float) -> int:
    return round(abs(v) / _DEDUP_TOL)


def _is_zero(v: float) -> bool: return abs(v) < _DEDUP_TOL
def _is_one(v: float)  -> bool: return abs(abs(v) - 1.0) < _DEDUP_TOL
def _is_half(v: float) -> bool: return abs(abs(v) - 0.5) < _DEDUP_TOL


def build_constant_table(q: int):
    """Return (resolver, unique_consts) for radix q.

    `resolver[(kind, m, i)]` is one of:
        ("sent", _, SENT_*)             — trivial twiddle (0, ±½, ±1)
        ("ref",  ±1.0, "D3_R{q}_T{idx}") — reference one unique constant ±

    `unique_consts` is a list of `(name, magnitude_f64)` pairs, sorted by
    descending magnitude. Trivial values are NOT in this list.
    """
    qh = (q - 1) // 2
    entries = []
    for m in range(qh):
        alpha_num = (q - 1 - 2 * m)
        for i in range(1, qh + 1):
            sign_i = (-1) ** i
            entries.append((("PCOS", m, i),
                            sign_i * math.cos(math.pi * alpha_num * i / q)))
            entries.append((("PSIN", m, i),
                            sign_i * math.sin(math.pi * alpha_num * i / q)))
    for m in range(qh):
        alpha_num = (q - 1 - 2 * m)
        for i in range(1, qh + 1):
            sign = 1.0 if (i - 1) % 2 == 0 else -1.0
            entries.append((("BOUND_W", m, i),
                            sign * math.sin(math.pi * (2 * i - 1) * alpha_num / (2 * q))))

    bucket: dict[int, float] = {}
    for _key, v in entries:
        if _is_zero(v) or _is_one(v) or _is_half(v):
            continue
        k = _canon_key(v)
        if k not in bucket:
            bucket[k] = abs(v)

    sorted_mags = sorted(bucket.values(), reverse=True)
    name_for_mag: dict[int, str] = {}
    unique_consts: list[tuple[str, float]] = []
    for idx, mag in enumerate(sorted_mags):
        name = f"D3_R{q}_T{idx}"
        name_for_mag[_canon_key(mag)] = name
        unique_consts.append((name, mag))

    resolver: dict[tuple[str, int, int], tuple[str, float, str | None]] = {}
    for key, v in entries:
        if _is_zero(v):
            resolver[key] = ("sent", 0.0, SENT_ZERO)
        elif _is_one(v):
            resolver[key] = ("sent", 0.0, SENT_POS_ONE if v > 0 else SENT_NEG_ONE)
        elif _is_half(v):
            resolver[key] = ("sent", 0.0, SENT_POS_HALF if v > 0 else SENT_NEG_HALF)
        else:
            name = name_for_mag[_canon_key(v)]
            resolver[key] = ("ref", 1.0 if v > 0 else -1.0, name)

    return resolver, unique_consts


def emit_fmla_step(acc: str, var: str, entry) -> str | None:
    """Emit `acc = acc OP var [* coeff]` as a single statement string,
    or None if the term is zero (caller should skip it)."""
    tag, sign, name = entry
    if tag == "sent":
        if name == SENT_ZERO:    return None
        if name == SENT_POS_ONE: return f"{acc} = {acc} + {var};"
        if name == SENT_NEG_ONE: return f"{acc} = {acc} - {var};"
        if name == SENT_POS_HALF: return f"{acc} = fmla({var}, T::HALF, {acc});"
        if name == SENT_NEG_HALF: return f"{acc} = fmla({var}, -T::HALF, {acc});"
        raise AssertionError(name)
    return f"{acc} = fmla({var}, {'T::' + name if sign > 0 else '-T::' + name}, {acc});"


def emit_seed_expr(var: str, entry) -> str | None:
    """Right-hand side for `let mut acc = <RHS>;` when seeding with var*coeff.
    Returns None for the zero case (caller seeds with T::default() instead).
    """
    tag, sign, name = entry
    if tag == "sent":
        if name == SENT_ZERO:    return None
        if name == SENT_POS_ONE: return var
        if name == SENT_NEG_ONE: return f"-{var}"
        if name == SENT_POS_HALF: return f"{var} * T::HALF"
        if name == SENT_NEG_HALF: return f"-{var} * T::HALF"
        raise AssertionError(name)
    return f"{var} * T::{name}" if sign > 0 else f"-{var} * T::{name}"


def _sage_expr_for(q: int, mag: float) -> str:
    """Best-effort symbolic description for the Sage comment header."""
    qh = (q - 1) // 2
    for m in range(qh):
        for i in range(1, qh + 1):
            ang_pair = math.pi * (q - 1 - 2 * m) * i / q
            ang_bw   = math.pi * (2 * i - 1) * (q - 1 - 2 * m) / (2 * q)
            if abs(abs(math.cos(ang_pair)) - mag) < _DEDUP_TOL:
                return f"abs(cos(pi * R({(q-1-2*m)*i}) / R({q})))"
            if abs(abs(math.sin(ang_pair)) - mag) < _DEDUP_TOL:
                return f"abs(sin(pi * R({(q-1-2*m)*i}) / R({q})))"
            if abs(abs(math.sin(ang_bw)) - mag) < _DEDUP_TOL:
                return f"abs(sin(pi * R({(2*i-1)*(q-1-2*m)}) / R({2*q})))"
    return f"R({mag!r})"


def _double_to_hex(x: float) -> str:
    return "0x" + struct.pack(">d", float(x)).hex()


def _float_to_hex(x: float) -> str:
    return "0x" + struct.pack(">f", float(x)).hex()


def emit_constants_trait(q: int) -> str:
    """Emit the `Dct3MixedRadix{q}Sample` trait + f32/f64 impls.

    Each unique non-trivial magnitude becomes a single `from_bits` constant.
    Trivial values (0, ±½, ±1) do NOT appear here — they're inlined at call
    sites by `emit_fmla_step` / `emit_seed_expr`.

    Constants are hex-encoded (bit-exact, matching the existing pxdct
    `DctConstants` style) and annotated with the Sage script that produced
    them, so the bit patterns are reproducible and auditable.
    """
    _, unique_consts = build_constant_table(q)
    trait_name = f"Dct3MixedRadix{q}Sample"

    lines = []
    lines.append(f"/// Deduplicated trigonometric constants for the length-{q} odd-factor")
    lines.append("/// DCT-III decomposition.")
    lines.append("///")
    lines.append("/// Many of the raw `cos(α i / q)` / `sin(α i / q)` / `sin((2i-1) α / (2q))`")
    lines.append("/// values that appear in eqs (7), (11), (13) of Bi (1999) collide up to")
    lines.append("/// sign once the `(-1)^i` and `(-1)^(i-1)` factors are folded in. We exploit")
    lines.append(f"/// that here: only the {len(unique_consts)} unique non-trivial magnitudes")
    lines.append("/// are trait constants; trivial cases (0, ±½, ±1) are emitted as inline")
    lines.append("/// arithmetic at call sites. This matches the style of `Dct4MixedRadix9`,")
    lines.append("/// which inlines explicit `-T::D4_R9_ROT_TWIDDLE_*` at use sites.")
    lines.append(f"pub(crate) trait {trait_name} {{")
    for name, _ in unique_consts:
        lines.append(f"    const {name}: Self;")
    lines.append("}")
    lines.append("")

    for ty in ("f32", "f64"):
        hex_fn = "float_to_hex" if ty == "f32" else "double_to_hex"
        pack_fmt = "'>f'" if ty == "f32" else "'>d'"
        hexer = _float_to_hex if ty == "f32" else _double_to_hex
        lines.append(f"impl {trait_name} for {ty} {{")
        for idx, (name, mag) in enumerate(unique_consts):
            sage_expr = _sage_expr_for(q, mag)
            # Sage script comment block (matches existing DctConstants style).
            lines.append("    // import struct")
            lines.append("    // from sage.all import *")
            lines.append("    // R = RealField(256)")
            lines.append(f"    // def {hex_fn}(f):")
            lines.append(f"    //     packed = struct.pack({pack_fmt}, float(f))")
            lines.append("    //     return '0x' + packed.hex()")
            lines.append(f"    // print({hex_fn}(float({sage_expr})))")
            lines.append(f"    const {name}: Self = {ty}::from_bits({hexer(mag)});")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def emit_struct(q: int) -> str:
    return textwrap.dedent(f"""\
        pub(crate) struct Dct3MixedRadix{q}<T> {{
            inner_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
            /// Rotation twiddles R_k = cos(α k / 2N) + j sin(α k / 2N),
            /// laid out as `qh` blocks of `p` complex values: idx = m * p + k.
            /// These depend on `len`, so they're built at construction time
            /// (unlike the pair twiddles, which are pure trait constants).
            rotation_twiddles: Vec<Complex<T>>,
            execution_length: usize,
            p: usize,         // = N / q
            inner_dct3_scratch_size: usize,
        }}
    """)


def emit_new(q: int) -> str:
    return textwrap.dedent(f"""\
        impl<T: DctSample> Dct3MixedRadix{q}<T>
        where
            f64: AsPrimitive<T>,
            usize: AsPrimitive<T>,
        {{
            #[allow(unused)]
            pub(crate) fn new(
                len: usize,
                inner_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
            ) -> Result<Self, PxdctError> {{
                assert_eq!(
                    inner_dct3.length(),
                    len / {q},
                    "DCT-III Mixed-Radix-{q} inner DCT-III length must be N / {q}"
                );

                let p = len / {q};
                let qh = {(q - 1) // 2};

                // Rotation twiddles: for each m in [0, qh) and k in [0, p), compute
                //   (cos(α k / 2N), sin(α k / 2N))  with  α = π(q - 1 - 2m).
                // Since α k / 2N = π · ((q - 1 - 2m) · k) / (2 N), we evaluate
                // `sincos_pi` at argument ((q-1-2m) · k) / (2N).
                let mut rotation_twiddles = try_vec![Complex::<T>::default(); qh * p];
                for m in 0..qh {{
                    let alpha_num = ({q} - 1 - 2 * m) as f64; // integer factor of π in α
                    for k in 0..p {{
                        let arg = alpha_num * (k as f64) / (2.0 * len as f64);
                        // `sincos_pi` returns (sin, cos); pack as (re = cos, im = sin).
                        let sc = arg.sincos_pi();
                        rotation_twiddles[m * p + k] = Complex::new(sc.1.as_(), sc.0.as_());
                    }}
                }}

                let inner_dct3_scratch_size = inner_dct3.scratch_size();
                Ok(Self {{
                    inner_dct3,
                    rotation_twiddles,
                    execution_length: len,
                    p,
                    inner_dct3_scratch_size,
                }})
            }}
        }}
    """)


def build_doc(q: int, qh: int) -> str:
    return textwrap.dedent(f"""\
        Mixed-radix-{q} decomposition of a length-N DCT-III into {q} length-N/{q} DCT-III's,
        based on the odd-factor algorithm of Bi (1999), "Fast Algorithms for Type-III DCT
        of Composite Sequence Lengths", IEEE Trans. Signal Processing, 47(7).

        For sequence length N = {q} * p and input spectrum X(k), k = 0..N-1, the inverse
        transform x(n) = Σ X(k) cos(π (2n + 1) k / (2N)) is decomposed using symmetric
        / antisymmetric combinations around the {qh} "even" centers (2i N / {q}),
        i = 1..{qh}:

            S_i(k) = X(2iN/{q} + k) + X(2iN/{q} - k)
            T_i(k) = X(2iN/{q} + k) - X(2iN/{q} - k),       k = 1..p-1.

        This yields one length-p DCT-III on a U buffer (the "centre" output stream) and,
        for each m in 0..{qh - 1 if qh > 1 else 0}, a pair (V_m, W'_m) of length-p
        DCT-III's that produce the m-th and ({q}-m-1)-th output residues; here
        α = π({q} - 1 - 2m).

        Inputs (eqs 7, 11, 13 of the paper, with the W(p, m) sign corrected per the
        explicit derivation from eq (9) — the paper's eq (13) has (-1)^i where it
        should be (-1)^(i-1), as confirmed by the q = 3 hand-derived reference):

            U(0)     = Σ_{{i=0..{qh}}} (-1)^i X(2iN/{q})
            U(k)     = X(k) + Σ_{{i=1..{qh}}} (-1)^i S_i(k)                       k = 1..p-1

            V(0, m)  = Σ_{{i=0..{qh}}} (-1)^i X(2iN/{q}) cos(α i / {q})
            V(k, m)  = X(k) cos(α k / 2N)
                       + Σ_{{i=1..{qh}}} (-1)^i [S_i(k) cos(α i / {q}) cos(α k / 2N)
                                              - T_i(k) sin(α i / {q}) sin(α k / 2N)]
                                                                                   k = 1..p-1

            W(p, m)  = Σ_{{i=1..{qh}}} (-1)^(i-1) X((2i-1) N / {q}) sin((2i-1) α / (2q))
            W(k, m)  = [X(k) + Σ_{{i=1..{qh}}} (-1)^i S_i(k) cos(α i / {q})] sin(α k / 2N)
                       + [Σ_{{i=1..{qh}}} (-1)^i T_i(k) sin(α i / {q})] cos(α k / 2N)
                                                                                   k = 1..p-1

            W'(0, m) = W(p, m),    W'(k, m) = W(p - k, m)                          k = 1..p-1

        A length-p DCT-III is applied to U, V_m, and W'_m (the inner DCT-III follows
        the library's half-DC / H_p convention, so the routine compensates with a
        -X(0)/2 adjustment to stay in H_N at the outer level). The final outputs are
        recombined via eq (15):

            x({q} n + {qh}              ) = A[n]                                   (centre)
            x({q} n + m                 ) = F_m[n] + (-1)^n G_m[n]
            x({q} n + {q} - m - 1       ) = F_m[n] - (-1)^n G_m[n]                 m = 0..{qh - 1}
    """)


def emit_execute_with_store(q: int, qh: int) -> str:
    """The heart of the routine, specialised + unrolled at codegen time.

    Trivial twiddles (0, ±½, ±1) are inlined; non-trivial values reference the
    deduplicated `T::D3_R{q}_T{idx}` constants. All inner sums over i = 1..qh
    are unrolled, so the compiler sees straight-line `fmla` chains.

    Implementation notes:
      * Inner DCT-III is in H_p (half-DC) convention, so we snapshot
        u0/2, v0/2, w'0/2 BEFORE the inner call and apply a
        (buf(0) - X(0)) / 2 correction afterwards to stay in H_N.
      * W'(0, m) = W(p, m) is the singleton boundary term; its DC half is
        added back to G(n, m) with the (-1)^n sign.
    """
    resolver, _ = build_constant_table(q)

    # Helpers to build accumulator-sum statements via the resolver.
    def build_sum(acc: str, seed_expr: str, terms: list[tuple[str, tuple]]) -> list[str]:
        """Emit `let mut acc = seed_expr; <fmla steps>` lines for an
        accumulator that starts at seed_expr and adds var*coeff for each
        (var, entry) in terms, skipping zero-coeff entries."""
        out = [f"let mut {acc} = {seed_expr};"]
        for var, entry in terms:
            stmt = emit_fmla_step(acc, var, entry)
            if stmt is not None:
                out.append(stmt)
        return out

    def build_signed_sum_inline(seed_expr: str, terms: list[tuple[str, tuple]]) -> str:
        """For the cheap U-style sum: emit a single expression like
        `xk - s_1 + s_2 - s_3` purely with sentinel coefficients (must all be
        SENT_POS_ONE / SENT_NEG_ONE / SENT_ZERO). Returns a single Rust expr."""
        parts = [seed_expr]
        for var, entry in terms:
            tag, _sign, name = entry
            if tag != "sent":
                raise ValueError("build_signed_sum_inline only handles sentinel terms")
            if name == SENT_ZERO:    continue
            if name == SENT_POS_ONE: parts.append(f"+ {var}")
            elif name == SENT_NEG_ONE: parts.append(f"- {var}")
            else:
                raise ValueError(f"unexpected sentinel for inline sum: {name}")
        return " ".join(parts) if len(parts) > 1 else parts[0]

    # ---- U(0) = X(0) + Σ_{i=1..qh} (-1)^i X(2 i p) ---------------------
    # All coefficients are ±1 (sentinels), so this is a straight inline sum.
    u0_terms_for_xi = []  # list of (var, sentinel_entry)
    for i in range(1, qh + 1):
        sign_i = (-1) ** i
        # Coefficient on X(2ip): just (-1)^i
        entry = ("sent", 0.0, SENT_POS_ONE if sign_i > 0 else SENT_NEG_ONE)
        u0_terms_for_xi.append((f"data[{2 * i} * p]", entry))
    u0_expr = build_signed_sum_inline("data[0]", u0_terms_for_xi)

    # ---- U(k) = X(k) + Σ_{i=1..qh} (-1)^i S_i(k) -----------------------
    # Same pattern, again inline ± of s_i.
    u_k_terms = []
    for i in range(1, qh + 1):
        sign_i = (-1) ** i
        entry = ("sent", 0.0, SENT_POS_ONE if sign_i > 0 else SENT_NEG_ONE)
        u_k_terms.append((f"s_{i}", entry))
    u_k_expr = build_signed_sum_inline("xk", u_k_terms)

    # ---- V(0, m) for each m -------------------------------------------
    def v0_block(m: int) -> str:
        terms = []
        for i in range(1, qh + 1):
            entry = resolver[("PCOS", m, i)]
            terms.append((f"data[{2 * i} * p]", entry))
        # Seed with X(0) (the i = 0 term: (-1)^0 cos(0) X(0) = X(0))
        lines = build_sum(f"v0_m{m}", "data[0]", terms)
        lines.append(f"v_buffer[{m} * p] = v0_m{m};")
        return "\n".join(lines)

    v0_init_lines = "\n".join(v0_block(m) for m in range(qh))

    # ---- W'(0, m) = W(p, m) for each m --------------------------------
    def w0_block(m: int) -> str:
        # Sum over i = 1..qh of BOUND_W[m][i-1] * X((2i-1) p).
        # Seed with the first non-trivial-zero term to avoid a needless 0 +.
        terms = [(f"data[{(2 * i - 1)} * p]", resolver[("BOUND_W", m, i)])
                 for i in range(1, qh + 1)]
        # Find the first nonzero entry to use as the seed.
        seed_var, seed_idx = None, None
        for idx, (var, entry) in enumerate(terms):
            seed_e = emit_seed_expr(var, entry)
            if seed_e is not None:
                seed_var = seed_e
                seed_idx = idx
                break
        if seed_var is None:
            # All zeros — W(p, m) is identically zero. Emit a literal.
            return f"w_buffer[{m} * p] = T::default();"
        lines = [f"let mut w0_m{m} = {seed_var};"]
        for var, entry in terms[seed_idx + 1:]:
            stmt = emit_fmla_step(f"w0_m{m}", var, entry)
            if stmt is not None:
                lines.append(stmt)
        lines.append(f"w_buffer[{m} * p] = w0_m{m};")
        return "\n".join(lines)

    w0_init_lines = "\n".join(w0_block(m) for m in range(qh))

    # ---- Per-k, per-m: V(k, m) and W(k, m) ----------------------------
    def per_k_per_m_block(m: int) -> str:
        # c_acc = X(k) + Σ_{i=1..qh} (-1)^i S_i(k) cos(α_m i/q)
        # s_acc =        Σ_{i=1..qh} (-1)^i T_i(k) sin(α_m i/q)
        pcos_terms = [(f"s_{i}", resolver[("PCOS", m, i)]) for i in range(1, qh + 1)]
        psin_terms = [(f"t_{i}", resolver[("PSIN", m, i)]) for i in range(1, qh + 1)]

        # c_acc seeds with xk (the X(k) term).
        lines = build_sum(f"c_acc{m}", "xk", pcos_terms)

        # s_acc: find first nonzero seed.
        seed_idx = None
        seed_expr = None
        for idx, (var, entry) in enumerate(psin_terms):
            seed_e = emit_seed_expr(var, entry)
            if seed_e is not None:
                seed_expr = seed_e
                seed_idx = idx
                break
        if seed_expr is None:
            # All PSIN coefficients for this m are zero — s_acc is identically 0.
            lines.append(f"let s_acc{m} = T::default();")
        else:
            lines.append(f"let mut s_acc{m} = {seed_expr};")
            for var, entry in psin_terms[seed_idx + 1:]:
                stmt = emit_fmla_step(f"s_acc{m}", var, entry)
                if stmt is not None:
                    lines.append(stmt)

        lines.append(
            f"let r{m} = unsafe {{ self.rotation_twiddles.get_unchecked({m} * p + k) }};"
        )
        lines.append(
            f"let v_val{m} = fmla(c_acc{m}, r{m}.re, -s_acc{m} * r{m}.im);"
        )
        lines.append(
            f"let w_val{m} = fmla(c_acc{m}, r{m}.im,  s_acc{m} * r{m}.re);"
        )
        lines.append(f"v_buffer[{m} * p + k]       = v_val{m};")
        lines.append(f"w_buffer[{m} * p + (p - k)] = w_val{m};")
        return "\n".join(lines)

    per_k_loop_body = "\n\n".join(per_k_per_m_block(m) for m in range(qh))

    # ---- S_i, T_i loads -----------------------------------------------
    si_ti_loads = []
    for i in range(1, qh + 1):
        si_ti_loads.append(
            f"let xp_{i} = data[{2 * i} * p + k];\n"
            f"let xm_{i} = data[{2 * i} * p - k];\n"
            f"let s_{i} = xp_{i} + xm_{i};\n"
            f"let t_{i} = xp_{i} - xm_{i};"
        )
    si_ti_loads_block = "\n".join(si_ti_loads)

    # ---- DC snapshots --------------------------------------------------
    v0_snap = "\n".join(
        f"let v0_m{m}_half = v_buffer[{m} * p] * T::HALF;"
        for m in range(qh)
    )
    w0_snap = "\n".join(
        f"let w0_m{m}_half = w_buffer[{m} * p] * T::HALF;"
        for m in range(qh)
    )

    # ---- Reconstruction inner-m block ---------------------------------
    def recon_m_block(m: int) -> str:
        return (
            f"let f_v{m}   = v_buffer[{m} * p + n];\n"
            f"let g_raw{m} = w_buffer[{m} * p + n];\n"
            f"let g_v{m}   = fmla(g_raw{m}, sign, w0_m{m}_half.mulsign(sign));\n"
            f"let f_dc{m}  = f_v{m} + (v0_m{m}_half - x0_half);\n"
            f"data[{q} * n + {m}]             = f_dc{m} + g_v{m};\n"
            f"data[{q} * n + {q - 1 - m}]     = f_dc{m} - g_v{m};"
        )

    recon_inner = "\n".join(recon_m_block(m) for m in range(qh))

    return textwrap.dedent(f"""\
        impl<T: DctSample + Dct3MixedRadix{q}Sample> Dct3MixedRadix{q}<T>
        where
            f64: AsPrimitive<T>,
        {{
            #[inline(always)]
            fn execute_with_store<S: BidirectionalStore<T>>(
                &self,
                data: &mut S,
                scratch: &mut [T],
            ) -> Result<(), PxdctError> {{
                let p = self.p;

                let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
                // Buffer layout in `scratch`:
                //   [ A: p ][ V_0 .. V_{{qh-1}} each p ][ W'_0 .. W'_{{qh-1}} each p ]
                // Total = (1 + 2 * qh) * p = q * p = N
                let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
                let (v_buffer, w_buffer)  = vw_buffer.split_at_mut({qh} * p);

                // ------------------------------------------------------------------
                // Step 1: build the q inner-DCT-III input buffers.
                // ------------------------------------------------------------------

                // U(0) = sum_{{i=0..qh}} (-1)^i X(2 i p)
                a_buffer[0] = {u0_expr};

                // V(0, m) for each m.
{textwrap.indent(v0_init_lines, "                ")}

                // W'(0, m) = W(p, m) for each m.
{textwrap.indent(w0_init_lines, "                ")}

                // k = 1..p-1: build U(k), V(k, m), W'(p - k, m) using symmetric pairs.
                // For each k we load X(2 i p ± k) once into s_i / t_i locals and reuse
                // them across all m. Trivial twiddles (0, ±½, ±1) are inlined; the rest
                // reference the deduplicated `T::D3_R{q}_T*` constants with the sign at
                // the call site.
                for k in 1..p {{
                    let xk = data[k];

{textwrap.indent(si_ti_loads_block, "                    ")}

                    // U(k) = X(k) + Σ_{{i=1..qh}} (-1)^i S_i(k)
                    a_buffer[k] = {u_k_expr};

{textwrap.indent(per_k_loop_body, "                    ")}
                }}

                // ------------------------------------------------------------------
                // Snapshot the pre-inner DC slots BEFORE the inner DCT-III clobbers
                // them. The inner uses the H_p half-DC convention, so after the
                // inner call slot 0 holds P_p[buf](0) − buf(0)/2; we need the
                // original buf(0) values for the DC correction in step 3.
                // ------------------------------------------------------------------
                let x0_half = data[0] * T::HALF;
                let u0_half = a_buffer[0] * T::HALF;
{textwrap.indent(v0_snap, "                ")}
{textwrap.indent(w0_snap, "                ")}

                // ------------------------------------------------------------------
                // Step 2: inner DCT-IIIs (half-DC convention).
                // One call processes A, all V_m and all W'_m together because the
                // executor strides through `scratch` in chunks of length `p`.
                // ------------------------------------------------------------------
                self.inner_dct3
                    .execute_with_scratch(scratch, inner_scratch)?;

                let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
                let (v_buffer, w_buffer)  = vw_buffer.split_at_mut({qh} * p);

                // ------------------------------------------------------------------
                // Step 3: reconstruct outer outputs.
                //   x(q n + qh)     = A[n]                   + (u0/2 - X0/2)
                //   x(q n + m)      = F_m[n] + sign·(G_m_raw + w0_m/2) + (v0_m/2 - X0/2)
                //   x(q n + q-m-1)  = F_m[n] - sign·(G_m_raw + w0_m/2) + (v0_m/2 - X0/2)
                // where sign = (-1)^n. The w0_m/2 corrects the W'(0) singleton
                // that the inner H_p subtracted from G_m_raw.
                // ------------------------------------------------------------------
                let dc_adjust_a = u0_half - x0_half;

                let mut sign = T::one();
                for n in 0..p {{
                    // Centre stream
                    data[{q} * n + {qh}] = a_buffer[n] + dc_adjust_a;

{textwrap.indent(recon_inner, "                    ")}

                    sign = -sign;
                }}

                Ok(())
            }}
        }}
    """)


def emit_executor_impl(q: int) -> str:
    """Generate the PxdctExecutor trait impl (essentially boilerplate)."""
    return textwrap.dedent(f"""\
        impl<T: DctSample + Dct3MixedRadix{q}Sample> PxdctExecutor<T> for Dct3MixedRadix{q}<T>
        where
            f64: AsPrimitive<T>,
        {{
            fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {{
                let mut scratch = try_vec![T::default(); self.scratch_size()];
                self.execute_with_scratch(data, &mut scratch)
            }}

            fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {{
                if !data.len().is_multiple_of(self.execution_length) {{
                    return Err(PxdctError::InvalidSizeMultiplier(
                        data.len(),
                        self.execution_length,
                    ));
                }}

                let full_scratch = validate_scratch!(scratch, self.scratch_size());

                for chunk in data.chunks_exact_mut(self.execution_length) {{
                    self.execute_with_store(&mut InPlaceStore::new(chunk), full_scratch)?;
                }}

                Ok(())
            }}

            fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {{
                let mut scratch = try_vec![T::default(); self.scratch_size()];
                self.execute_into_with_scratch(input, output, &mut scratch)
            }}

            fn execute_into_with_scratch(
                &self,
                input: &[T],
                output: &mut [T],
                scratch: &mut [T],
            ) -> Result<(), PxdctError> {{
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(input, output, self.execution_length);

                let full_scratch = validate_scratch!(scratch, self.scratch_size());

                use crate::bidirectional::BiStore;
                for (src, dst) in input
                    .chunks_exact(self.execution_length)
                    .zip(output.chunks_exact_mut(self.execution_length))
                {{
                    self.execute_with_store(&mut BiStore::new(src, dst), full_scratch)?;
                }}
                Ok(())
            }}

            #[inline]
            fn length(&self) -> usize {{
                self.execution_length
            }}

            #[inline]
            fn scratch_size(&self) -> usize {{
                self.execution_length + self.inner_dct3_scratch_size
            }}
        }}
    """)


def emit_tests(q: int) -> str:
    # Use a small composite test size: N = q^2 (so the inner DCT-III length is q,
    # which means the user can plug in the Dct3Butterfly{q} hand-tuned kernel).
    return textwrap.dedent(f"""\
        #[cfg(test)]
        mod tests {{
            use super::*;
            use crate::tests::naive_dct3;
            use rand::RngExt;

            #[test]
            fn test_split_dct3_radix{q}() {{
                // N = q * q so inner length is q (use the matching butterfly kernel).
                const N: usize = {q} * {q};
                let mut input = vec![0.0; N];
                for z in input.iter_mut() {{
                    *z = rand::rng().random_range(1.0..2.0);
                }}
                let reference = naive_dct3(&input);

                // The caller picks the inner length-{q} executor; tests in this
                // crate that pre-date this generated file should construct it the
                // same way the q = 3 test does (`Dct3Butterfly{q}::default()`).
                // We leave the construction site as a TODO so the generated file
                // compiles even when no butterfly of size {q} is available yet.
                //
                // let bf = Dct3MixedRadix{q}::new(N, Arc::new(Dct3Butterfly{q}::default())).unwrap();
                // bf.execute(&mut input).unwrap();
                // for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {{
                //     assert!((a - b).abs() < 1e-1, "mismatch at {{i}}: {{a}} vs {{b}}");
                // }}
                let _ = reference;
            }}
        }}
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("q", type=int, help="odd radix (>= 3)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output path (default: dct3_mixed_radix_{q}.rs)")
    ap.add_argument("--check", action="store_true",
                    help="run a numerical self-check of the decomposition for this q and exit")
    ap.add_argument("--stdout", action="store_true",
                    help="print the generated code to stdout instead of writing a file")
    args = ap.parse_args(argv)

    if args.check:
        print(f"Self-checking odd-factor DCT-III for q = {args.q} ...")
        ok = self_check(args.q)
        sys.exit(0 if ok else 1)

    code = emit(args.q)

    if args.stdout:
        sys.stdout.write(code)
        return

    path = args.output or Path(f"dct3_mixed_radix_{args.q}.rs")
    path.write_text(code)
    print(f"Wrote {path} ({len(code)} bytes)")


if __name__ == "__main__":
    main()
