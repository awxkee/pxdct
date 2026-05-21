#!/usr/bin/env python3
"""
Automatic DST-VII butterfly matrix deriver and Rust code generator.

DST-VII definition (matching the existing codebase convention):
  Y[k] = sum_{n=0}^{N-1} x[n] * sin((n+1)(2k+1) * pi / (2N+1))

for k = 0..N-1.

Unique twiddle constants: sin(m*pi/(2N+1)) for m=1..floor((2N+1)/2),
using sin((2N+1-m)*pi/(2N+1)) = sin(m*pi/(2N+1)).
Signs come from reducing the angle into [0, pi] or (pi, 2pi).
Zero entries (sin = 0) are handled by omitting the fmla term.

Optimisations
-------------
Two orthogonal optimisations are applied and their savings compose.

1. Row pre-sum optimisation (applied to residual inputs)
   When gcd(2k+1, 2N+1) = g > 1, row k has only (2N+1)/(2g) distinct
   sine values.  Within the residual inputs (not in any column group),
   inputs sharing a coefficient are pre-summed before the single multiply.
   Pre-sums that appear in more than one row are hoisted to named
   let-bindings (e.g. let r0 = x0 + x9;) and reused by name.

2. Column-group optimisation
   For each prime p dividing (2N+1), the input columns where (n+1) is
   divisible by p (the "Gp group") produce only floor((2N+1)/(2p))
   distinct dot-products across all N output rows.  These are computed
   once and routed (with +/- signs) to every output row.

   N=16  (period 33 = 3x11):  246 -> 152 multiplications (-38%)
   N=32  (period 65 = 5x13): 1000 -> 656 multiplications (-34%)
   N=8   (period 17, prime):   no column groups; row pre-sum only
"""

import math
import sys


# ── angle reduction ──────────────────────────────────────────────────────────

def reduce_angle(raw_index, period):
    m = raw_index % (2 * period)
    if m == 0:
        return (0, 1)
    sign = 1
    if m > period:
        m = 2 * period - m
        sign = -1
    half = period // 2
    if m > half:
        m = period - m
    return (m, sign)


# ── matrix builder ───────────────────────────────────────────────────────────

def build_matrix(N):
    period = 2 * N + 1
    return [
        [reduce_angle((n + 1) * (2 * k + 1), period) for n in range(N)]
        for k in range(N)
    ]


def get_unique_indices(matrix):
    return sorted({idx for row in matrix for idx, _ in row if idx != 0})


# ── prime factorisation ──────────────────────────────────────────────────────

def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


# ── group-expression canonical form ─────────────────────────────────────────

def _canonical_key(members):
    """
    Canonical form for a set of (sign, n) pairs: the frozenset whose member
    with the smallest input index n carries a positive sign.  A group and its
    global negation share the same canonical key.
    """
    key = frozenset(members)
    neg = frozenset([(-s, n) for s, n in members])
    min_sign = min(key, key=lambda x: x[1])[0]
    return key if min_sign > 0 else neg


def _group_rust_expr(canonical, negated=False):
    """
    Build a Rust expression string for a canonical group of inputs.
    e.g. frozenset({(1,0),(1,9)}) -> 'x0 + x9'
         with negated=True        -> '-x0 - x9'   (used in seed line only)
    """
    members = sorted(canonical, key=lambda x: x[1])
    pos = [n for s, n in members if s > 0]
    neg = [n for s, n in members if s < 0]
    if not negated:
        parts = [f"x{n}" for n in pos] + [f"-x{n}" for n in neg]
    else:
        # flip every sign
        parts = [f"x{n}" for n in neg] + [f"-x{n}" for n in pos]
    return " + ".join(parts).replace("+ -", "- ")


# ── hoisted pre-sum registry ─────────────────────────────────────────────────

def collect_hoisted_presums(matrix, residual_inputs):
    """
    Scan every output row for multi-element residual groups.  Any group
    expression that appears (as itself or its negation) in more than one row
    is a candidate for hoisting to a named let-binding.

    Returns an ordered dict: canonical_key -> name  (e.g. 'r0', 'r1', …)
    Also returns a list of (name, rust_expr) definition strings in order.
    """
    freq = {}   # canonical_key -> occurrence count
    for row in matrix:
        seen_in_row = set()
        groups = {}
        for n in residual_inputs:
            idx, sign = row[n]
            if idx == 0:
                continue
            groups.setdefault(idx, []).append((sign, n))
        for members in groups.values():
            if len(members) < 2:
                continue
            ck = _canonical_key(members)
            if ck not in seen_in_row:
                freq[ck] = freq.get(ck, 0) + 1
                seen_in_row.add(ck)

    # Only hoist if used more than once.
    hoisted = {}   # canonical_key -> name
    defs    = []   # [(name, rust_expr), ...]
    counter = 0
    for ck, count in sorted(freq.items(), key=lambda x: -x[1]):
        if count < 2:
            continue
        name = f"r{counter}"
        counter += 1
        hoisted[ck] = name
        defs.append((name, _group_rust_expr(ck)))

    return hoisted, defs


# ── column-group analysis ────────────────────────────────────────────────────

def find_column_groups(N):
    """
    For each prime p dividing (2N+1), identify the Gp column group
    (inputs n where (n+1) % p == 0) and determine:

      'inputs'       - list of input indices in the group
      'canon_rows'   - list of row indices whose Gp dot-product is canonical
      'canon_coeffs' - for each canonical row, the list of (sine_idx, sign)
      'routing'      - dict  k -> (canon_index, +-1)  or  k -> None  (zero)
    """
    period = 2 * N + 1
    matrix = build_matrix(N)
    primes = _prime_factors(period)
    s_raw  = [math.sin((i + 1) * math.pi / period) for i in range(N)]

    import random as _rnd
    rng = _rnd.Random(0xDEADBEEF ^ N)

    result = {}
    for p in sorted(primes):
        gp = [n for n in range(N) if (n + 1) % p == 0]
        if not gp:
            continue

        probe = [rng.gauss(0, 1) for _ in gp]

        row_dot = []
        for k in range(N):
            val = sum(
                (s_raw[matrix[k][gp[i]][0] - 1] if matrix[k][gp[i]][0] != 0 else 0.0)
                * matrix[k][gp[i]][1]
                * probe[i]
                for i in range(len(gp))
            )
            row_dot.append(val)

        canon_rows = []
        routing    = {}

        for k in range(N):
            if abs(row_dot[k]) < 1e-12:
                routing[k] = None
                continue
            found = False
            for ci, ref in enumerate(canon_rows):
                if abs(row_dot[k] - row_dot[ref]) < 1e-10:
                    routing[k] = (ci, +1); found = True; break
                if abs(row_dot[k] + row_dot[ref]) < 1e-10:
                    routing[k] = (ci, -1); found = True; break
            if not found:
                routing[k] = (len(canon_rows), +1)
                canon_rows.append(k)

        canon_coeffs = [
            [(matrix[ref_k][gp[j]][0], matrix[ref_k][gp[j]][1]) for j in range(len(gp))]
            for ref_k in canon_rows
        ]

        result[p] = {
            'inputs':       gp,
            'canon_rows':   canon_rows,
            'canon_coeffs': canon_coeffs,
            'routing':      routing,
        }

    return result


# ── Rust code builders ───────────────────────────────────────────────────────

def _s_ref(s_array_idx, sign):
    """'self.s[i]' or '-self.s[i]'."""
    r = f"self.s[{s_array_idx}]"
    return f"-{r}" if sign < 0 else r


def _input_expr(members, hoisted):
    """
    Return the Rust expression for a group of (sign, n) members.
    If the group (or its negation) is in *hoisted*, return (name, is_negated).
    Otherwise return (inline_expr, False).

    The returned expression is safe to use directly in fmla() without extra parens.
    For the seed multiply  expr * coeff  the caller wraps if the expr contains
    spaces and needs parens (handled in build_fmla_chain).
    """
    if len(members) == 1:
        s, n = members[0]
        return (f"x{n}", s < 0)   # is_negated means we must negate the coeff

    ck = _canonical_key(members)
    actual = frozenset(members)
    neg_ck = frozenset([(-s, n) for s, n in ck])
    is_neg = (actual == neg_ck)

    if ck in hoisted:
        name = hoisted[ck]
        return (name, is_neg)

    # Inline — no parens needed in fmla position.
    return (_group_rust_expr(ck, negated=is_neg), False)


def _needs_parens_for_multiply(expr):
    """True if *expr* must be wrapped in parens to appear before '*'."""
    # If the expression starts with '-' or contains spaces it binds loosely.
    return expr.startswith("-") or " " in expr


def build_fmla_chain(var_name, terms, idx_to_s):
    """
    Emit a flat sequential fmla chain.

    terms: list of (input_expr, s_array_idx, sign, is_input_negated)
      - input_expr      : Rust expression for the input (xN, rK, or inline sum)
      - s_array_idx     : index into self.s  (-1 = skip/zero term)
      - sign            : coefficient sign (+1 or -1)
      - is_input_negated: True when the input_expr itself should be sign-flipped
                          relative to the canonical sign already embedded in sign

    Returns list of code-line strings (no leading spaces).
    """
    # Filter zero terms.
    valid = [(e, s, sgn, neg) for e, s, sgn, neg in terms if s >= 0]
    if not valid:
        return [f"let {var_name} = T::zero();"]

    lines = []
    e0, s0, sgn0, neg0 = valid[0]
    coeff0 = f"self.s[{s0}]"
    # Combined sign: sgn0 XOR neg0
    effective_neg = (sgn0 < 0) ^ neg0
    if _needs_parens_for_multiply(e0):
        mul_expr = f"({e0}) * {coeff0}"
    else:
        mul_expr = f"{e0} * {coeff0}"
    if effective_neg:
        lines.append(f"let {var_name} = -({mul_expr});")
    else:
        lines.append(f"let {var_name} = {mul_expr};")

    for e, s, sgn, neg in valid[1:]:
        coeff = _s_ref(s, sgn)
        # If the input is negated, flip the coefficient sign.
        if neg:
            coeff = _s_ref(s, -sgn)
        lines.append(f"let {var_name} = fmla({e}, {coeff}, {var_name});")

    return lines


def build_colgroup_dp_lines(dp_name, gp_inputs, coeffs, idx_to_s):
    """Flat fmla chain for one canonical column-group dot-product."""
    terms = [
        (f"x{gp_inputs[i]}", idx_to_s[coeffs[i][0]], coeffs[i][1], False)
        for i in range(len(gp_inputs))
        if coeffs[i][0] != 0
    ]
    return build_fmla_chain(dp_name, terms, idx_to_s)


def build_residual_lines(k, row, residual_inputs, idx_to_s, var_name, hoisted):
    """
    Flat fmla chain for the residual contribution to output row k.
    Uses hoisted pre-sum names where available.
    """
    groups = {}
    for n in residual_inputs:
        idx, sign = row[n]
        if idx == 0:
            continue
        groups.setdefault(idx, []).append((sign, n))

    terms = []
    for idx in sorted(groups.keys()):
        members = groups[idx]
        expr, is_neg = _input_expr(members, hoisted)
        terms.append((expr, idx_to_s[idx], +1, is_neg))

    return build_fmla_chain(var_name, terms, idx_to_s)


def _assembly_line(var_name, acc_name, cg_terms):
    """
    Build the final assembly line:  let y{k} = acc + dp_a - dp_b + ...
    cg_terms: list of (dp_var_name, sign)
    Avoids '+ (-dp)' by using subtraction directly.
    """
    expr = acc_name
    for dp, sgn in cg_terms:
        if sgn > 0:
            expr += f" + {dp}"
        else:
            expr += f" - {dp}"
    return f"let {var_name} = {expr};"


# ── display helpers ──────────────────────────────────────────────────────────

def print_matrix(matrix, N):
    period = 2 * N + 1
    unique = get_unique_indices(matrix)
    print(f"\nDST-VII N={N} matrix  (Sk = sin(k*pi/{period}))")
    print(f"Unique constants: S{', S'.join(str(i) for i in unique)}")
    print()
    print("     " + "  ".join(f"  x{n}" for n in range(N)))
    for k, row in enumerate(matrix):
        cells = []
        for idx, sign in row:
            if idx == 0:
                cells.append("  0 ")
            else:
                cells.append(f"{'+'if sign>0 else '-'}S{idx}")
        print(f"Y{k} [ " + "  ".join(cells) + " ]")
    print()


def _op_count(N, cgroups, residual_inputs):
    matrix    = build_matrix(N)
    orig_muls = sum(sum(1 for idx, _ in row if idx != 0) for row in matrix)
    opt_muls  = 0
    for info in cgroups.values():
        for coeffs in info['canon_coeffs']:
            opt_muls += sum(1 for idx, _ in coeffs if idx != 0)
    for row in matrix:
        seen = set()
        for n in residual_inputs:
            idx, _ = row[n]
            if idx != 0:
                seen.add(idx)
        opt_muls += len(seen)
    return orig_muls, opt_muls


# ── full Rust code generator ─────────────────────────────────────────────────

def generate_rust(N):
    period   = 2 * N + 1
    matrix   = build_matrix(N)
    unique   = get_unique_indices(matrix)
    idx_to_s = {idx: i for i, idx in enumerate(unique)}
    num_c    = len(unique)
    cgroups  = find_column_groups(N)

    grouped_col_inputs = set()
    for info in cgroups.values():
        grouped_col_inputs.update(info['inputs'])
    residual_inputs = [n for n in range(N) if n not in grouped_col_inputs]

    orig_muls, opt_muls = _op_count(N, cgroups, residual_inputs)

    # Collect hoisted pre-sum names for repeated group expressions.
    hoisted, hoisted_defs = collect_hoisted_presums(matrix, residual_inputs)

    lines = []

    # ── struct ──────────────────────────────────────────────────────────────
    lines += [
        f"#[derive(Debug, Clone)]",
        f"pub(crate) struct Dst7Butterfly{N}<T: DctSample> {{",
        f"    s: [T; {num_c}],",
        f"}}",
        "",
    ]

    # ── Default ─────────────────────────────────────────────────────────────
    lines += [
        f"impl<T: DctSample> Default for Dst7Butterfly{N}<T>",
        f"where",
        f"    f64: AsPrimitive<T>,",
        f"{{",
        f"    fn default() -> Self {{",
        f"        Self {{",
        f"            s: [",
    ]
    for idx in unique:
        lines.append(
            f"                ({idx}.0 / {period}.0).sinpi().as_(),"
            f" // S{idx} = sin({idx}*pi/{period})"
        )
    lines += [
        f"            ],",
        f"        }}",
        f"    }}",
        f"}}",
        "",
    ]

    # ── exec impl ───────────────────────────────────────────────────────────
    lines += [
        f"impl<T: DctSample> Dst7Butterfly{N}<T>",
        f"where",
        f"    f64: AsPrimitive<T>,",
        f"{{",
        f"    #[inline(always)]",
        f"    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {{",
    ]

    for n in range(N):
        lines.append(f"        let x{n} = data[{n}];")
    lines.append("")

    # Matrix comment.
    lines.append(f"        // Analytically reduced from a {period}-point RDFT")
    lines.append(f"        // Sk = sin(k*pi/{period}), s[k-1] = Sk")
    for k, row in enumerate(matrix):
        ts = []
        for n, (idx, sign) in enumerate(row):
            if idx == 0: continue
            ts.append(f"{'+'if sign>0 else '-'}S{idx}*x{n}")
        lines.append(f"        //   Y{k} = " + " ".join(ts))
    lines.append("")

    # ── Column-group canonical dot-products ─────────────────────────────────
    if cgroups:
        lines.append(
            f"        // Column-group + row pre-sum optimisation:"
            f" {orig_muls} -> {opt_muls} multiplications"
            f" (saving {orig_muls - opt_muls})."
        )
        lines.append(
            f"        // For each prime p | {period}, inputs where (n+1)%p==0"
            f" form group Gp whose dot-products repeat across all rows."
        )
        lines.append("")

        for p, info in sorted(cgroups.items()):
            gp      = info['inputs']
            n_canon = len(info['canon_rows'])
            lines.append(
                f"        // G{p} ({{{', '.join(f'x{n}' for n in gp)}}})"
                f"  ->  {n_canon} canonical dot-product(s)"
            )
            for ci, (ref_k, coeffs) in enumerate(
                    zip(info['canon_rows'], info['canon_coeffs'])):
                dp_name = f"dp{p}_{ci}"
                lines.append(f"        // dp{p}_{ci}: row {ref_k} Gp pattern")
                for stmt in build_colgroup_dp_lines(dp_name, gp, coeffs, idx_to_s):
                    lines.append(f"        {stmt}")
            lines.append("")

    # ── Hoisted pre-sum variables ────────────────────────────────────────────
    if hoisted_defs:
        lines.append(
            f"        // Hoisted pre-sums reused across multiple rows."
        )
        for name, expr in hoisted_defs:
            lines.append(f"        let {name} = {expr};")
        lines.append("")

    # ── Per-row residual + assembly ──────────────────────────────────────────
    for k, row in enumerate(matrix):
        has_residual = any(row[n][0] != 0 for n in residual_inputs)

        cg_terms = []
        for p, info in sorted(cgroups.items()):
            r = info['routing'][k]
            if r is None:
                continue
            ci, sign = r
            cg_terms.append((f"dp{p}_{ci}", sign))

        if not has_residual and not cg_terms:
            lines.append(f"        let y{k} = T::zero();")
            lines.append(f"        data[{k}] = y{k};")
            lines.append("")
            continue

        if not has_residual:
            # Row is a pure sum/difference of column-group dot-products.
            first_dp, first_sgn = cg_terms[0]
            expr = first_dp if first_sgn > 0 else f"-{first_dp}"
            for dp, sgn in cg_terms[1:]:
                expr += f" {'+'if sgn>0 else '-'} {dp}"
            lines.append(f"        let y{k} = {expr};")
            lines.append(f"        data[{k}] = y{k};")
            lines.append("")
            continue

        acc_var   = f"acc{k}"
        res_stmts = build_residual_lines(
            k, row, residual_inputs, idx_to_s, acc_var, hoisted
        )
        for stmt in res_stmts:
            lines.append(f"        {stmt}")

        if not cg_terms:
            # Rename last acc line to y{k}.
            # The last emitted statement assigned to acc_var; rename it.
            last = lines[-1]
            # Could be a fmla line or the seed line.
            lines[-1] = last.replace(
                f"let {acc_var} = fmla(", f"let y{k} = fmla(", 1
            ).replace(
                f"let {acc_var} = -(", f"let y{k} = -(", 1
            )
            # Handle pure seed (single term, no fmla).
            if f"let {acc_var} =" in lines[-1]:
                lines[-1] = lines[-1].replace(f"let {acc_var} =", f"let y{k} =", 1)
        else:
            lines.append(f"        {_assembly_line(f'y{k}', acc_var, cg_terms)}")

        lines.append(f"        data[{k}] = y{k};")
        lines.append("")

    lines += [
        f"    }}",
        f"}}",
        "",
        f"define_in_place_butterfly!(Dst7Butterfly{N}, {N});",
    ]

    return "\n".join(lines)


# ── verification ─────────────────────────────────────────────────────────────

def verify_matrix(N):
    import random
    period   = 2 * N + 1
    matrix   = build_matrix(N)
    unique   = get_unique_indices(matrix)
    idx_to_s = {idx: i for i, idx in enumerate(unique)}
    s_vals   = [math.sin(idx * math.pi / period) for idx in unique]
    random.seed(42 + N)
    x = [random.gauss(0, 1) for _ in range(N)]
    y_matrix = [
        sum(sign * s_vals[idx_to_s[idx]] * x[n]
            for n, (idx, sign) in enumerate(row) if idx != 0)
        for row in matrix
    ]
    y_naive = [
        sum(x[n] * math.sin((n + 1) * (2 * k + 1) * math.pi / period)
            for n in range(N))
        for k in range(N)
    ]
    ok = all(abs(a - b) < 1e-12 for a, b in zip(y_matrix, y_naive))
    print(f"Matrix derivation  N={N}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        for k in range(N):
            d = abs(y_matrix[k] - y_naive[k])
            if d > 1e-12:
                print(f"  Y{k}: matrix={y_matrix[k]:.8f}  naive={y_naive[k]:.8f}"
                      f"  diff={d:.2e}")
    return ok


def verify_optimised(N):
    import random
    period   = 2 * N + 1
    matrix   = build_matrix(N)
    unique   = get_unique_indices(matrix)
    idx_to_s = {idx: i for i, idx in enumerate(unique)}
    s_vals   = [math.sin(idx * math.pi / period) for idx in unique]
    cgroups  = find_column_groups(N)

    grouped_col_inputs = set()
    for info in cgroups.values():
        grouped_col_inputs.update(info['inputs'])
    residual = [n for n in range(N) if n not in grouped_col_inputs]
    hoisted, _ = collect_hoisted_presums(matrix, residual)

    all_ok = True
    for trial in range(200):
        random.seed(trial * 17 + N)
        x = [random.gauss(0, 1) for _ in range(N)]

        y_ref = [
            sum(sign * s_vals[idx_to_s[idx]] * x[n]
                for n, (idx, sign) in enumerate(row) if idx != 0)
            for row in matrix
        ]

        dp_vals = {}
        for p, info in cgroups.items():
            gp = info['inputs']
            for ci, (ref_k, coeffs) in enumerate(
                    zip(info['canon_rows'], info['canon_coeffs'])):
                dp_vals[(p, ci)] = sum(
                    (s_vals[idx_to_s[coeffs[j][0]]] if coeffs[j][0] != 0 else 0.0)
                    * coeffs[j][1] * x[gp[j]]
                    for j in range(len(gp))
                )

        y_opt = []
        for k, row in enumerate(matrix):
            res_groups = {}
            for n in residual:
                idx, sign = row[n]
                if idx == 0: continue
                res_groups.setdefault(idx, []).append((sign, n))
            acc = sum(
                sum(sign * x[n] for sign, n in members) * s_vals[idx_to_s[idx]]
                for idx, members in res_groups.items()
            )
            for p, info in cgroups.items():
                r = info['routing'][k]
                if r is None: continue
                ci, sign = r
                acc += sign * dp_vals[(p, ci)]
            y_opt.append(acc)

        for k in range(N):
            if abs(y_opt[k] - y_ref[k]) > 1e-9:
                print(f"  FAIL N={N} trial={trial} row={k}: "
                      f"opt={y_opt[k]:.8f}  ref={y_ref[k]:.8f}")
                all_ok = False

    print(f"Combined optimisation N={N}: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
    if len(sys.argv) > 1:
        sizes = [int(a) for a in sys.argv[1:]]

    for N in sizes:
        print("=" * 72)
        print(f"DST-VII Butterfly  N={N}  (2N+1={2*N+1})")
        print("=" * 72)

        if not verify_matrix(N):
            print(f"ERROR: matrix derivation failed for N={N}, skipping.")
            continue
        if not verify_optimised(N):
            print(f"ERROR: combined optimisation incorrect for N={N}, skipping.")
            continue

        print_matrix(build_matrix(N), N)
        print(generate_rust(N))
        print()


if __name__ == "__main__":
    main()