#!/usr/bin/env python3
"""
Automatic DCT-VII butterfly matrix deriver and Rust code generator.

DCT-VII definition (matching the existing codebase convention):
  Y[k] = (1/2)*x[0] + sum_{n=1}^{N-1} x[n] * cos(n*(2k+1)*pi/(2N-1))

for k = 0..N-1, M = 2N-1.

The x[0] column is always a fixed 1/2 factor, seeded as x0 * T::HALF.

Unique twiddle constants: cos(m*pi/M) for m in [1, floor(M/2)].
Using cos((M-m)*pi/M) = -cos(m*pi/M) for sign folding.
idx=0 means cos(0)=1 (a plain ±1 add/subtract, no multiply).

optimizations
-------------
Mirrors the two optimizations from the DST-VII generator.

1. Row pre-sum optimization (residual inputs, n>=1 only)
   When gcd(2k+1, M) = g > 1, row k has fewer distinct cosine values.
   Residual n>=1 inputs that share a coefficient in the same row are
   pre-summed.  idx=0 inputs (coefficient ±1) are summed and added/
   subtracted directly without a multiply.  Pre-sums used in more than
   one row are hoisted to named let-bindings.

2. Column-group optimization
   For each prime p | M, inputs n where p | n (n >= 1) form group Gp.
   These produce only floor(M/(2p)) distinct dot-products across all rows.
   Each is computed once and routed (±) to every row that uses it.
   Savings compose with the row pre-sum applied to the residual inputs.

   N=8  (M=15=3x5): 45 -> 26 multiplications (-42%)
"""

import math
import sys


# ── angle reduction ──────────────────────────────────────────────────────────

def reduce_cos(raw, M):
    """
    Given raw such that the entry is cos(raw*pi/M), return (idx, sign)
    where entry = sign * cos(idx*pi/M), idx in [0, M//2].

    idx=0 means cos(0)=1 (a ±1 entry, no multiply needed).
    """
    m = raw % (2 * M)
    sign = 1
    if m > M:
        m = 2 * M - m          # period fold, no sign change
    if m > M // 2:
        m = M - m
        sign = -1
    return (m, sign)


# ── matrix builder ───────────────────────────────────────────────────────────

def build_matrix(N):
    """
    Build the N x N DCT-VII matrix.

    Row k, column n entry:
      n=0:  ('half', 1)           always x[0] * 0.5
      n>=1: ('trig', idx, sign)   idx=0 means ±1, idx>0 means ±cos(idx*pi/M)
    """
    M = 2 * N - 1
    matrix = []
    for k in range(N):
        row = [('half', 1)]
        for n in range(1, N):
            idx, sign = reduce_cos(n * (2 * k + 1), M)
            row.append(('trig', idx, sign))
        matrix.append(row)
    return matrix


def get_unique_trig_indices(matrix):
    """Return sorted list of unique idx > 0 used in the matrix."""
    return sorted({e[1] for row in matrix for e in row
                   if e[0] == 'trig' and e[1] > 0})


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
    with the smallest n carries a positive sign.  A group and its negation
    share the same canonical key.
    """
    key = frozenset(members)
    neg = frozenset([(-s, n) for s, n in members])
    return key if min(key, key=lambda x: x[1])[0] > 0 else neg


def _group_rust_expr(canonical):
    """Build 'x1 + x4 - x7' style expression from canonical frozenset."""
    members = sorted(canonical, key=lambda x: x[1])
    pos = [n for s, n in members if s > 0]
    neg = [n for s, n in members if s < 0]
    parts = [f"x{n}" for n in pos] + [f"-x{n}" for n in neg]
    return " + ".join(parts).replace("+ -", "- ")


# ── hoisted pre-sum registry ─────────────────────────────────────────────────

def collect_hoisted_presums(matrix, residual_n):
    """
    Scan every row for multi-element residual groups (n >= 1 only, grouped
    by cosine index).  Groups appearing in more than one row are hoisted
    to named let-bindings.

    Returns (hoisted_dict, defs_list):
      hoisted_dict: canonical_key -> name  (e.g. 'r0')
      defs_list:    [(name, rust_expr), ...]
    """
    freq = {}
    for row in matrix:
        seen = set()
        groups = {}
        for n in residual_n:
            e = row[n]
            if e[0] != 'trig':
                continue
            idx, sign = e[1], e[2]
            groups.setdefault(idx, []).append((sign, n))
        for members in groups.values():
            if len(members) < 2:
                continue
            ck = _canonical_key(members)
            if ck not in seen:
                freq[ck] = freq.get(ck, 0) + 1
                seen.add(ck)

    hoisted = {}
    defs    = []
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
    For each prime p | (2N-1), find group Gp = {n : 1 <= n <= N-1, p | n}.

    Returns dict p -> {
      'inputs':       [n, ...],
      'canon_rows':   [k, ...],
      'canon_coeffs': [[(idx, sign), ...], ...],   one list per canonical row
      'routing':      {k: (canon_idx, +-1) or None}
    }

    canon_coeffs uses idx=-1 as a sentinel for zero (shouldn't occur for n>=1
    with p prime, but included for safety).
    """
    M = 2 * N - 1
    matrix = build_matrix(N)
    primes = _prime_factors(M)

    import random as _rnd
    rng = _rnd.Random(0xDEADBEEF ^ N)

    c_raw = [math.cos(i * math.pi / M) for i in range(N + 1)]

    def entry_float(e):
        if e[0] == 'half':  return 0.5
        _, idx, sign = e
        return sign * (c_raw[idx] if idx <= N else 0.0)

    result = {}
    for p in sorted(primes):
        gp = [n for n in range(1, N) if n % p == 0]
        if not gp:
            continue

        probe    = [rng.gauss(0, 1) for _ in gp]
        row_dot  = [
            sum(entry_float(matrix[k][gp[i]]) * probe[i] for i in range(len(gp)))
            for k in range(N)
        ]

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

        canon_coeffs = []
        for ref_k in canon_rows:
            coeffs = []
            for j in range(len(gp)):
                e = matrix[ref_k][gp[j]]
                if e[0] == 'trig':
                    coeffs.append((e[1], e[2]))   # (idx, sign)
                else:
                    coeffs.append((-1, 1))         # sentinel zero
            canon_coeffs.append(coeffs)

        result[p] = {
            'inputs':       gp,
            'canon_rows':   canon_rows,
            'canon_coeffs': canon_coeffs,
            'routing':      routing,
        }
    return result


# ── Rust code builders ───────────────────────────────────────────────────────

def _c_ref(c_array_idx, sign):
    """'self.c[i]' or '-self.c[i]'."""
    r = f"self.c[{c_array_idx}]"
    return f"-{r}" if sign < 0 else r


def _needs_parens_for_multiply(expr):
    """True if expr must be parenthesised before '*'."""
    return expr.startswith("-") or " " in expr


def build_colgroup_dp_lines(dp_name, gp, coeffs, idx_to_c):
    """
    Flat fmla chain for one canonical column-group dot-product.
    coeffs: list of (idx, sign) per group input; idx=0 means ±1.
    """
    terms = [(gp[i], coeffs[i][0], coeffs[i][1])
             for i in range(len(gp)) if coeffs[i][0] >= 0]

    if not terms:
        return [f"let {dp_name} = T::zero();"]

    lines = []
    n0, idx0, sgn0 = terms[0]

    if idx0 == 0:
        # ±1 seed: no multiply
        if sgn0 > 0:
            lines.append(f"let {dp_name} = x{n0};")
        else:
            lines.append(f"let {dp_name} = -x{n0};")
    else:
        mul = f"x{n0} * self.c[{idx_to_c[idx0]}]"
        lines.append(f"let {dp_name} = {'-(' + mul + ')' if sgn0 < 0 else mul};")

    for n, idx, sgn in terms[1:]:
        if idx == 0:
            op = "+" if sgn > 0 else "-"
            lines.append(f"let {dp_name} = {dp_name} {op} x{n};")
        else:
            coeff = _c_ref(idx_to_c[idx], sgn)
            lines.append(f"let {dp_name} = fmla(x{n}, {coeff}, {dp_name});")

    return lines


def _resolve_group(members, hoisted):
    """
    Return (expr, is_negated) for a group of (sign, n) inputs.
    Uses a hoisted name if available, else an inline expression.
    is_negated means the expression should be negated relative to canonical sign.
    """
    if len(members) == 1:
        s, n = members[0]
        return (f"x{n}", s < 0)

    ck      = _canonical_key(members)
    neg_ck  = frozenset([(-s, n) for s, n in ck])
    is_neg  = (frozenset(members) == neg_ck)

    if ck in hoisted:
        return (hoisted[ck], is_neg)

    # inline — use the canonical polarity (is_neg drives sign in fmla coeff)
    return (_group_rust_expr(ck), is_neg)


def build_residual_lines(k, row, residual_n, idx_to_c, var_name, hoisted):
    """
    Flat let-chain for the residual (non-Gp) contribution to row k.
    Seed is always x0 * T::HALF (n=0 column).
    Then residual n>=1 columns are grouped by cosine index.
    idx=0 groups (coefficient ±1) become plain add/subtract lines.
    """
    lines = [f"let {var_name} = h0;"]

    # Group residual n>=1 inputs by cosine index.
    groups = {}
    for n in residual_n:
        e = row[n]
        if e[0] != 'trig':
            continue
        idx, sign = e[1], e[2]
        groups.setdefault(idx, []).append((sign, n))

    sorted_idxs = sorted(groups.keys())

    if not sorted_idxs:
        # No residual trig terms; just x0*T::HALF.
        lines[-1] = lines[-1].replace(f"let {var_name}", f"let {var_name}")
        return lines

    for i, idx in enumerate(sorted_idxs):
        members = groups[idx]
        is_last = (i == len(sorted_idxs) - 1)
        lhs = var_name   # we always keep the same name (no rename for residual)

        expr, is_neg = _resolve_group(members, hoisted)

        if idx == 0:
            # Coefficient ±1: build directly from actual members to avoid
            # sign errors when the canonical form is negated.
            pos_ns = sorted([n for s, n in members if s > 0])
            neg_ns = sorted([n for s, n in members if s < 0])
            pos_part = " + ".join(f"x{n}" for n in pos_ns)
            neg_part = " - ".join(f"x{n}" for n in neg_ns)
            if pos_part and neg_part:
                rhs = f"{pos_part} - {neg_part}"
            elif pos_part:
                rhs = pos_part
            else:
                rhs = f"-x{neg_ns[0]}" + "".join(f" - x{n}" for n in neg_ns[1:])
            lines.append(f"let {lhs} = {lhs} + {rhs};")
        else:
            coeff = _c_ref(idx_to_c[idx], -1 if is_neg else +1)
            if _needs_parens_for_multiply(expr) and not expr.startswith("r") and not expr.startswith("x"):
                lines.append(f"let {lhs} = fmla(({expr}), {coeff}, {lhs});")
            else:
                lines.append(f"let {lhs} = fmla({expr}, {coeff}, {lhs});")

    return lines


def _assembly_line(var_name, acc_name, cg_terms):
    """Build 'let y{k} = acc ± dp_a ± dp_b ...' without '+ (-dp)' patterns."""
    expr = acc_name
    for dp, sgn in cg_terms:
        expr += f" {'+'if sgn>0 else '-'} {dp}"
    return f"let {var_name} = {expr};"


# ── display helpers ──────────────────────────────────────────────────────────

def _entry_str(e):
    if e[0] == 'half': return '+H'
    _, idx, sign = e
    s = '+' if sign > 0 else '-'
    return f'{s}1' if idx == 0 else f'{s}C{idx}'


def print_matrix(matrix, N):
    M = 2 * N - 1
    unique = get_unique_trig_indices(matrix)
    print(f"\nDCT-VII N={N} matrix  (Ck = cos(k*pi/{M}), H = 1/2)")
    print(f"Unique trig constants: C{', C'.join(str(i) for i in unique)}")
    print()
    print("     " + "  ".join(f"  x{n}" for n in range(N)))
    for k, row in enumerate(matrix):
        print(f"Y{k} [ " + "  ".join(_entry_str(e) for e in row) + " ]")
    print()


def _op_count(N, matrix, cgroups, residual_n):
    """Return (orig_muls, opt_muls) counting only genuine trig multiplications."""
    orig = sum(sum(1 for e in row if e[0]=='trig' and e[1]>0) for row in matrix)

    opt = 0
    # Column-group canonicals.
    for info in cgroups.values():
        for coeffs in info['canon_coeffs']:
            opt += sum(1 for idx, _ in coeffs if idx > 0)
    # Residual: one mul per distinct idx>0 per row.
    for row in matrix:
        seen = set()
        for n in residual_n:
            e = row[n]
            if e[0] == 'trig' and e[1] > 0:
                seen.add(e[1])
        opt += len(seen)
    return orig, opt


# ── full Rust code generator ─────────────────────────────────────────────────

def generate_rust(N):
    M        = 2 * N - 1
    matrix   = build_matrix(N)
    unique   = get_unique_trig_indices(matrix)
    idx_to_c = {idx: i for i, idx in enumerate(unique)}
    num_c    = len(unique)
    cgroups  = find_column_groups(N)

    grouped_n = set()
    for info in cgroups.values():
        grouped_n.update(info['inputs'])
    residual_n = [n for n in range(1, N) if n not in grouped_n]

    orig_muls, opt_muls = _op_count(N, matrix, cgroups, residual_n)
    hoisted, hoisted_defs = collect_hoisted_presums(matrix, residual_n)

    lines = []

    # ── struct ──────────────────────────────────────────────────────────────
    lines += [
        f"#[derive(Debug, Clone)]",
        f"pub(crate) struct Dct7Butterfly{N}<T: DctSample> {{",
        f"    c: [T; {num_c}],",
        f"}}",
        "",
    ]

    # ── Default ─────────────────────────────────────────────────────────────
    lines += [
        f"impl<T: DctSample> Default for Dct7Butterfly{N}<T>",
        f"where",
        f"    f64: AsPrimitive<T>,",
        f"{{",
        f"    fn default() -> Self {{",
        f"        Self {{",
        f"            c: [",
    ]
    for idx in unique:
        lines.append(
            f"                ({idx}.0 / {M}.0).cospi().as_(),"
            f" // C{idx} = cos({idx}*pi/{M})"
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
        f"impl<T: DctSample> Dct7Butterfly{N}<T>",
        f"where",
        f"    f64: AsPrimitive<T>,",
        f"{{",
        f"    #[inline(always)]",
        f"    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {{",
    ]

    for n in range(N):
        lines.append(f"        let x{n} = data[{n}];")
    lines.append("")

    lines.append(f"        // Analytically reduced from a {4*N-2}-point RDFT")
    lines.append(f"        // H = 1/2, Ck = cos(k*pi/{M}), c[k-1] = Ck")
    for k, row in enumerate(matrix):
        lines.append(f"        //   Y{k} = " +
                     " ".join(f"{_entry_str(e)}*x{n}" for n, e in enumerate(row)))
    lines.append("")

    # ── Column-group canonical dot-products ─────────────────────────────────
    if cgroups:
        lines.append(
            f"        // Column-group + row pre-sum optimization:"
            f" {orig_muls} -> {opt_muls} multiplications"
            f" (saving {orig_muls - opt_muls})."
        )
        lines.append(
            f"        // For each prime p | {M}, inputs n where p|n form group Gp."
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
                for stmt in build_colgroup_dp_lines(dp_name, gp, coeffs, idx_to_c):
                    lines.append(f"        {stmt}")
            lines.append("")

    # ── Hoisted pre-sums ────────────────────────────────────────────────────
    if hoisted_defs:
        lines.append(f"        // Hoisted pre-sums reused across multiple rows.")
        for name, expr in hoisted_defs:
            lines.append(f"        let {name} = {expr};")
        lines.append("")

    # x0 * T::HALF is the seed for every row; compute it once to avoid
    # repeating floating-point work and ensure a single canonical value.
    lines.append(f"        let h0 = x0 * T::HALF;")
    lines.append("")

    # ── Per-row code ────────────────────────────────────────────────────────
    for k, row in enumerate(matrix):
        # n=0 always exists as HALF seed; check if any residual n>=1 terms.
        has_residual_trig = any(
            row[n][0] == 'trig' for n in residual_n
        ) if residual_n else False
        # n=0 is always present so residual block always emits at least the seed.

        cg_terms = []
        for p, info in sorted(cgroups.items()):
            r = info['routing'][k]
            if r is None:
                continue
            ci, sign = r
            cg_terms.append((f"dp{p}_{ci}", sign))

        acc_var   = f"acc{k}"
        res_stmts = build_residual_lines(
            k, row, residual_n, idx_to_c, acc_var, hoisted
        )
        for stmt in res_stmts:
            lines.append(f"        {stmt}")

        if not cg_terms:
            # Rename last line's LHS to y{k}.
            last = lines[-1]
            for old in [f"let {acc_var} = fmla(",
                        f"let {acc_var} = {acc_var} +",
                        f"let {acc_var} = {acc_var} -",
                        f"let {acc_var} ="]:
                if old in last:
                    lines[-1] = last.replace(old, old.replace(f"let {acc_var}", f"let y{k}"), 1)
                    break
        else:
            lines.append(f"        {_assembly_line(f'y{k}', acc_var, cg_terms)}")

        # Collapse "let acc{k} = seed; let y{k} = acc{k} + ..." into one line.
        if len(lines) >= 2:
            seed_line = f"        let {acc_var} = h0;"
            y_line = lines[-1]
            if lines[-2] == seed_line and f"= {acc_var} " in y_line:
                lines[-2] = y_line.replace(f"= {acc_var} ", "= h0 ", 1)
                lines.pop()

        lines.append(f"        data[{k}] = y{k};")
        lines.append("")

    lines += [
        f"    }}",
        f"}}",
        "",
        f"define_in_place_butterfly!(Dct7Butterfly{N}, {N});",
    ]

    return "\n".join(lines)


# ── verification ─────────────────────────────────────────────────────────────

def verify_matrix(N):
    import random
    M      = 2 * N - 1
    matrix = build_matrix(N)
    random.seed(137 + N)
    x = [random.gauss(0, 1) for _ in range(N)]

    def eval_entry(e, xn):
        if e[0] == 'half': return xn * 0.5
        _, idx, sign = e
        return sign * (1.0 if idx == 0 else math.cos(idx * math.pi / M)) * xn

    y_mat   = [sum(eval_entry(e, x[n]) for n, e in enumerate(row)) for row in matrix]
    y_naive = [
        0.5 * x[0] + sum(x[n] * math.cos(n * (2*k+1) * math.pi / M) for n in range(1, N))
        for k in range(N)
    ]
    ok = all(abs(a - b) < 1e-12 for a, b in zip(y_mat, y_naive))
    print(f"Matrix derivation  N={N}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        for k in range(N):
            d = abs(y_mat[k] - y_naive[k])
            if d > 1e-12:
                print(f"  Y{k}: matrix={y_mat[k]:.8f}  naive={y_naive[k]:.8f}  diff={d:.2e}")
    return ok


def verify_optimised(N):
    import random
    M        = 2 * N - 1
    matrix   = build_matrix(N)
    unique   = get_unique_trig_indices(matrix)
    idx_to_c = {idx: i for i, idx in enumerate(unique)}
    c_vals   = {idx: math.cos(idx * math.pi / M) for idx in unique}
    cgroups  = find_column_groups(N)

    grouped_n  = set()
    for info in cgroups.values(): grouped_n.update(info['inputs'])
    residual_n = [n for n in range(1, N) if n not in grouped_n]
    hoisted, _ = collect_hoisted_presums(matrix, residual_n)

    all_ok = True
    for trial in range(200):
        random.seed(trial * 17 + N)
        x = [random.gauss(0, 1) for _ in range(N)]

        y_ref = [
            0.5 * x[0] + sum(
                x[n] * math.cos(n * (2*k+1) * math.pi / M)
                for n in range(1, N)
            )
            for k in range(N)
        ]

        # Canonical column-group dot-products.
        dp_vals = {}
        for p, info in cgroups.items():
            gp = info['inputs']
            for ci, (ref_k, coeffs) in enumerate(
                    zip(info['canon_rows'], info['canon_coeffs'])):
                dp_vals[(p, ci)] = sum(
                    # (cos value, always positive) * sign * x[n]
                    (c_vals[coeffs[j][0]] if coeffs[j][0] > 0 else 1.0)
                    * coeffs[j][1] * x[gp[j]]
                    for j in range(len(gp))
                )

        y_opt = []
        for k, row in enumerate(matrix):
            acc = 0.5 * x[0]
            # Residual with pre-summing.
            res_groups = {}
            for n in residual_n:
                e = row[n]
                if e[0] != 'trig': continue
                res_groups.setdefault(e[1], []).append((e[2], n))
            for idx, members in res_groups.items():
                gs = sum(sign * x[n] for sign, n in members)
                acc += gs * (1.0 if idx == 0 else c_vals[idx])
            # Column-group routing.
            for p, info in cgroups.items():
                r = info['routing'][k]
                if r is None: continue
                ci, sign = r
                acc += sign * dp_vals[(p, ci)]
            y_opt.append(acc)

        for k in range(N):
            if abs(y_opt[k] - y_ref[k]) > 1e-9:
                print(f"  FAIL N={N} trial={trial} k={k}: "
                      f"opt={y_opt[k]:.8f}  ref={y_ref[k]:.8f}")
                all_ok = False

    print(f"Combined optimization N={N}: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
    if len(sys.argv) > 1:
        sizes = [int(a) for a in sys.argv[1:]]

    for N in sizes:
        print("=" * 72)
        print(f"DCT-VII Butterfly  N={N}  (2N-1={2*N-1})")
        print("=" * 72)

        if not verify_matrix(N):
            print(f"ERROR: matrix derivation failed for N={N}, skipping.")
            continue
        if not verify_optimised(N):
            print(f"ERROR: combined optimization incorrect for N={N}, skipping.")
            continue

        print_matrix(build_matrix(N), N)
        print(generate_rust(N))
        print()


if __name__ == "__main__":
    main()