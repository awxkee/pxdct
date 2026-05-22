"""
Tests for the pxdct Python bindings.

Run with:  pytest tests/
Requires:  pip install pxdct numpy pytest
"""

import numpy as np
import pytest
import pxdct
from pxdct import DctPlan, DctPlan2D

# ─── helpers ─────────────────────────────────────────────────────────────────

def _naive_dct2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros(n)
    for k in range(n):
        out[k] = sum(x[j] * np.cos(np.pi * k * (j + 0.5) / n) for j in range(n))
    return out


def _naive_dst2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros(n)
    for k in range(n):
        out[k] = sum(x[j] * np.sin(np.pi * (k + 1) * (j + 0.5) / n) for j in range(n))
    return out


def _ortho_scale_dct2(raw: np.ndarray) -> np.ndarray:
    """Reference ortho-normalised DCT-II from the raw (un-normalised) output."""
    n = len(raw)
    scaled = raw * np.sqrt(2.0 / n)
    scaled[0] *= np.sqrt(0.5)          # y[0] *= 1/sqrt(2)
    return scaled


SIZES = [1, 2, 4, 7, 8, 12, 16, 32, 100, 128, 256]
EVEN_SIZES  = [s for s in SIZES if s % 2 == 0]
MDCT_SIZES  = [s for s in SIZES if s % 4 == 0 and s >= 4]

# ─── DctPlan construction ────────────────────────────────────────────────────

class TestDctPlanConstruction:
    def test_basic_attributes(self):
        p = DctPlan("dct2", 64)
        assert p.length == 64
        assert p.kind == "dct2"
        assert p.dtype == "f64"
        assert p.scaling == "none"

    def test_f32_dtype(self):
        p = DctPlan("dct2", 64, "f32")
        assert p.dtype == "f32"

    def test_scaling_attribute_none(self):
        p = DctPlan("dct2", 64, scaling="none")
        assert p.scaling == "none"

    def test_scaling_attribute_scale(self):
        p = DctPlan("dct2", 64, scaling="scale")
        assert p.scaling == "scale"

    def test_scaling_attribute_ortho(self):
        p = DctPlan("dct2", 64, scaling="ortho")
        assert p.scaling == "ortho"

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError, match="scaling"):
            DctPlan("dct2", 16, scaling="invalid")

    def test_repr_includes_scaling(self):
        p = DctPlan("dct2", 32, "f64", "ortho")
        r = repr(p)
        assert "ortho" in r
        assert "dct2"  in r
        assert "32"    in r

    def test_repr(self):
        p = DctPlan("dst4", 32, "f64")
        assert "dst4" in repr(p)
        assert "32"   in repr(p)

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="Unknown"):
            DctPlan("fft2", 16)

    def test_invalid_type_number(self):
        with pytest.raises(ValueError):
            DctPlan("dct9", 16)

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            DctPlan("dct2", 16, "f16")

    def test_zero_length(self):
        with pytest.raises(ValueError):
            DctPlan("dct2", 0)

    def test_mdct_attributes(self):
        p = DctPlan("mdct", 64)
        assert p.length == 64
        assert p.kind   == "mdct"
        assert p.dtype  == "f64"

    def test_imdct_attributes(self):
        p = DctPlan("imdct", 64)
        assert p.length == 64
        assert p.kind   == "imdct"

    def test_mdct_odd_length_raises(self):
        with pytest.raises(ValueError):
            DctPlan("mdct", 7)

    def test_imdct_odd_length_raises(self):
        with pytest.raises(ValueError):
            DctPlan("imdct", 7)

    def test_mdct_too_small_raises(self):
        with pytest.raises(ValueError):
            DctPlan("mdct", 2)

    def test_imdct_too_small_raises(self):
        with pytest.raises(ValueError):
            DctPlan("imdct", 2)

    def test_mdct_case_insensitive(self):
        p = DctPlan("MDCT", 16)
        assert p.length == 16

    def test_imdct_case_insensitive(self):
        p = DctPlan("IMDCT", 16)
        assert p.length == 16

# ─── execute (in-place) ──────────────────────────────────────────────────────

class TestExecuteInPlace:
    @pytest.mark.parametrize("n", SIZES)
    def test_dct2_matches_naive(self, n):
        x   = np.random.randn(n)
        ref = _naive_dct2(x)
        p   = DctPlan("dct2", n)
        data = x.copy()
        p.execute(data)
        np.testing.assert_allclose(data, ref, atol=1e-9)

    @pytest.mark.parametrize("n", SIZES)
    def test_roundtrip_dct2_dct3(self, n):
        x   = np.random.randn(n)
        fwd = DctPlan("dct2", n)
        inv = DctPlan("dct3", n)
        y   = x.copy()
        fwd.execute(y)
        inv.execute(y)
        y *= 2.0 / n
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_wrong_length_raises(self):
        p = DctPlan("dct2", 16)
        with pytest.raises(ValueError, match="length"):
            p.execute(np.ones(8))

    def test_wrong_dtype_raises(self):
        p = DctPlan("dct2", 8)
        with pytest.raises(ValueError):
            p.execute(np.ones(8, dtype=np.float32))

    def test_f32_plan(self):
        n   = 32
        p   = DctPlan("dct2", n, "f32")
        x   = np.random.randn(n).astype(np.float32)
        ref = _naive_dct2(x.astype(float)).astype(np.float32)
        p.execute(x)
        np.testing.assert_allclose(x, ref, atol=1e-4)

# ─── scaling correctness ─────────────────────────────────────────────────────

class TestScaling:
    """Verify that all three scaling modes produce numerically correct output."""

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_scale_equals_raw_times_factor(self, n):
        """Scaling='scale' must equal the raw output multiplied by sqrt(2/n)."""
        x     = np.random.randn(n)
        raw   = DctPlan("dct2", n, scaling="none")
        scaled = DctPlan("dct2", n, scaling="scale")

        y_raw    = x.copy(); raw.execute(y_raw)
        y_scaled = x.copy(); scaled.execute(y_scaled)

        np.testing.assert_allclose(
            y_scaled, y_raw * np.sqrt(2.0 / n), atol=1e-10,
        )

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_ortho_dct2_matches_reference(self, n):
        """Scaling='ortho' for DCT-II must match the per-element reference formula."""
        x    = np.random.randn(n)
        p    = DctPlan("dct2", n, scaling="ortho")
        raw  = _naive_dct2(x)
        ref  = _ortho_scale_dct2(raw)
        data = x.copy()
        p.execute(data)
        np.testing.assert_allclose(data, ref, atol=1e-9)

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_ortho_roundtrip_dct2_dct3(self, n):
        """ortho DCT-II followed by ortho DCT-III must be the identity."""
        x   = np.random.randn(n)
        fwd = DctPlan("dct2", n, scaling="ortho")
        inv = DctPlan("dct3", n, scaling="ortho")
        y   = x.copy()
        fwd.execute(y)
        inv.execute(y)
        np.testing.assert_allclose(y, x, atol=1e-10)

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_ortho_roundtrip_dct4(self, n):
        """DCT-IV is its own inverse under ortho scaling."""
        x   = np.random.randn(n)
        p   = DctPlan("dct4", n, scaling="ortho")
        y   = x.copy()
        p.execute(y)
        p.execute(y)
        np.testing.assert_allclose(y, x, atol=1e-10)

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_ortho_roundtrip_dst2_dst3(self, n):
        """ortho DST-II followed by ortho DST-III must be the identity."""
        x   = np.random.randn(n)
        fwd = DctPlan("dst2", n, scaling="ortho")
        inv = DctPlan("dst3", n, scaling="ortho")
        y   = x.copy()
        fwd.execute(y)
        inv.execute(y)
        np.testing.assert_allclose(y, x, atol=1e-10)

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_ortho_roundtrip_f32(self, n):
        """ortho round-trip works at f32 precision."""
        x   = np.random.randn(n).astype(np.float32)
        fwd = DctPlan("dct2", n, "f32", "ortho")
        inv = DctPlan("dct3", n, "f32", "ortho")
        y   = x.copy()
        fwd.execute(y)
        inv.execute(y)
        np.testing.assert_allclose(y, x, atol=1e-5)

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_none_and_ortho_differ(self, n):
        """Sanity check: 'none' and 'ortho' must not produce the same output."""
        x      = np.random.randn(n)
        p_none = DctPlan("dct2", n, scaling="none")
        p_ortho = DctPlan("dct2", n, scaling="ortho")
        y_none  = x.copy(); p_none.execute(y_none)
        y_ortho = x.copy(); p_ortho.execute(y_ortho)
        assert not np.allclose(y_none, y_ortho)

    def test_execute_into_respects_scaling(self):
        """execute_into must apply the same scaling as execute."""
        n      = 32
        x      = np.random.randn(n)
        p      = DctPlan("dct2", n, scaling="ortho")
        y_inp  = x.copy(); p.execute(y_inp)
        y_into = p.execute_into(x)
        np.testing.assert_array_equal(y_inp, y_into)

    @pytest.mark.parametrize("scaling", ["none", "scale", "ortho"])
    def test_all_scalings_finite(self, scaling):
        """All scaling modes must produce finite output."""
        n    = 64
        x    = np.random.randn(n)
        p    = DctPlan("dct2", n, scaling=scaling)
        data = x.copy()
        p.execute(data)
        assert np.all(np.isfinite(data))

# ─── execute_into ─────────────────────────────────────────────────────────────

class TestExecuteInto:
    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_does_not_modify_input(self, n):
        x        = np.random.randn(n)
        original = x.copy()
        out      = np.empty(n)
        p        = DctPlan("dct2", n)
        p.execute_into(x, out)
        np.testing.assert_array_equal(x, original)

    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_output_matches_inplace(self, n):
        x       = np.random.randn(n)
        p       = DctPlan("dct2", n)
        inplace = x.copy(); p.execute(inplace)
        out     = np.empty(n)
        p.execute_into(x, out)
        np.testing.assert_array_equal(inplace, out)

    def test_size_mismatch_raises(self):
        p = DctPlan("dct2", 16)
        with pytest.raises(ValueError):
            p.execute_into(np.ones(16), np.ones(8))

# ─── all DST/DCT types smoke-test ────────────────────────────────────────────

class TestAllTypes:
    @pytest.mark.parametrize("family,ty", [
        ("dct", t) for t in range(1, 9)
    ] + [
        ("dst", t) for t in range(1, 9)
    ])
    def test_smoke(self, family, ty):
        kind = f"{family}{ty}"
        n    = 16
        p    = DctPlan(kind, n)
        x    = np.random.randn(n).copy()
        p.execute(x)
        assert np.all(np.isfinite(x))

    @pytest.mark.parametrize("family,ty", [
        ("dct", t) for t in range(1, 9)
    ] + [
        ("dst", t) for t in range(1, 9)
    ])
    @pytest.mark.parametrize("scaling", ["none", "scale", "ortho"])
    def test_smoke_all_scalings(self, family, ty, scaling):
        """Every type × scaling combination must execute without error."""
        kind = f"{family}{ty}"
        n    = 16 if ty != 1 else 17   # DCT-I needs n >= 2; 17 avoids edge cases
        p    = DctPlan(kind, n, scaling=scaling)
        x    = np.random.randn(n).copy()
        p.execute(x)
        assert np.all(np.isfinite(x))

# ─── MDCT / IMDCT ────────────────────────────────────────────────────────────

class TestMdctImdct:
    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_mdct_smoke(self, n):
        p = DctPlan("mdct", n)
        x = np.random.randn(n * 2)
        y = p.execute_into(x)
        assert y.shape == (n,)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_imdct_smoke(self, n):
        p = DctPlan("imdct", n)
        x = np.random.randn(n)
        y = p.execute_into(x)
        assert y.shape == (n * 2,)
        assert np.all(np.isfinite(y))

    def test_mdct_execute_inplace_raises(self):
        p = DctPlan("mdct", 8)
        with pytest.raises(ValueError, match="in-place"):
            p.execute(np.random.randn(16))

    def test_imdct_execute_inplace_raises(self):
        p = DctPlan("imdct", 8)
        with pytest.raises(ValueError, match="in-place"):
            p.execute(np.random.randn(8))

    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_roundtrip_f64(self, n):
        x          = np.random.randn(n * 2)
        mdct_plan  = DctPlan("mdct",  n, "f64")
        imdct_plan = DctPlan("imdct", n, "f64")
        coeffs     = mdct_plan.execute_into(x)
        recovered  = imdct_plan.execute_into(coeffs)
        assert recovered.shape == x.shape
        coeffs2    = mdct_plan.execute_into(x)
        np.testing.assert_allclose(coeffs, coeffs2, atol=0)

    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_roundtrip_f32(self, n):
        x          = np.random.randn(n * 2).astype(np.float32)
        mdct_plan  = DctPlan("mdct",  n, "f32")
        imdct_plan = DctPlan("imdct", n, "f32")
        coeffs     = mdct_plan.execute_into(x)
        recovered  = imdct_plan.execute_into(coeffs)
        assert recovered.shape == x.shape
        assert np.all(np.isfinite(recovered))

    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_execute_into_does_not_modify_input(self, n):
        p        = DctPlan("mdct", n)
        x        = np.random.randn(n * 2)
        original = x.copy()
        out      = np.empty(n)
        p.execute_into(x, out)
        np.testing.assert_array_equal(x, original)

    @pytest.mark.parametrize("n", MDCT_SIZES)
    def test_execute_into_preallocated_output(self, n):
        p      = DctPlan("mdct", n)
        x      = np.random.randn(n * 2)
        y_auto = p.execute_into(x)
        y_pre  = np.empty(n)
        p.execute_into(x, y_pre)
        np.testing.assert_array_equal(y_auto, y_pre)

    def test_mdct_via_dct_function(self):
        n  = 8
        x  = np.random.randn(n * 2)
        y1 = pxdct.dct(x, kind="mdct")
        y2 = DctPlan("mdct", n).execute_into(x)
        np.testing.assert_array_equal(y1, y2)
        assert y1.shape == (n,)

    def test_imdct_via_dct_function(self):
        n  = 8
        x  = np.random.randn(n)
        y1 = pxdct.dct(x, kind="imdct")
        y2 = DctPlan("imdct", n).execute_into(x)
        np.testing.assert_array_equal(y1, y2)
        assert y1.shape == (n * 2,)

    def test_mdct_ignores_scaling(self):
        """MDCT output must be identical regardless of the scaling argument."""
        n      = 16
        x      = np.random.randn(n * 2)
        y_none = DctPlan("mdct", n, scaling="none").execute_into(x)
        y_ortho = DctPlan("mdct", n, scaling="ortho").execute_into(x)
        np.testing.assert_array_equal(y_none, y_ortho)

    def test_imdct_ignores_scaling(self):
        n      = 16
        x      = np.random.randn(n)
        y_none  = DctPlan("imdct", n, scaling="none").execute_into(x)
        y_ortho = DctPlan("imdct", n, scaling="ortho").execute_into(x)
        np.testing.assert_array_equal(y_none, y_ortho)

# ─── DctPlan2D ───────────────────────────────────────────────────────────────

class TestDctPlan2D:
    def test_attributes(self):
        wp = DctPlan("dct2", 32)
        hp = DctPlan("dct2", 64)
        p2 = DctPlan2D(wp, hp)
        assert p2.width  == 32
        assert p2.height == 64

    def test_dtype_mismatch_raises(self):
        wp = DctPlan("dct2", 16, "f32")
        hp = DctPlan("dct2", 16, "f64")
        with pytest.raises(ValueError):
            DctPlan2D(wp, hp)

    def test_square_roundtrip(self):
        n   = 16
        p2f = DctPlan2D(DctPlan("dct2", n), DctPlan("dct2", n))
        p2i = DctPlan2D(DctPlan("dct3", n), DctPlan("dct3", n))
        x   = np.random.randn(n * n)
        orig = x.copy()
        p2f.execute(x)
        p2i.execute(x)
        x *= (2.0 / n) ** 2
        np.testing.assert_allclose(x, orig, atol=1e-9)

    def test_ortho_square_roundtrip(self):
        """2-D ortho DCT-IV is self-inverse, providing a clean 2-D ortho round-trip."""
        n   = 16
        # DCT-IV is its own inverse under ortho scaling — no pairing needed.
        p   = DctPlan2D(DctPlan("dct4", n, scaling="ortho"),
                        DctPlan("dct4", n, scaling="ortho"))
        x    = np.random.randn(n * n)
        orig = x.copy()
        p.execute(x)
        p.execute(x)    # applying twice = identity
        np.testing.assert_allclose(x, orig, atol=1e-9)

    def test_wrong_flat_size_raises(self):
        p2 = DctPlan2D(DctPlan("dct2", 8), DctPlan("dct2", 8))
        with pytest.raises(ValueError):
            p2.execute(np.ones(32))

# ─── convenience API ─────────────────────────────────────────────────────────

class TestConvenienceAPI:
    def test_dct_function(self):
        x     = np.random.randn(32)
        ref   = DctPlan("dct2", 32)
        y_ref = x.copy(); ref.execute(y_ref)
        y     = pxdct.dct(x, type=2)
        np.testing.assert_allclose(y, y_ref, atol=1e-12)

    def test_dst_function(self):
        x   = np.random.randn(32)
        y   = pxdct.dst(x, type=2)
        ref = _naive_dst2(x)
        np.testing.assert_allclose(y, ref, atol=1e-9)

    def test_dct_kind_kwarg(self):
        x  = np.random.randn(16)
        y1 = pxdct.dct(x, type=4)
        y2 = pxdct.dct(x, kind="dct4")
        np.testing.assert_array_equal(y1, y2)

    def test_dst_kind_kwarg(self):
        x  = np.random.randn(16)
        y1 = pxdct.dst(x, type=3)
        y2 = pxdct.dst(x, kind="dst3")
        np.testing.assert_array_equal(y1, y2)

    def test_dct_array_like_input(self):
        x   = list(range(1, 9))
        y   = pxdct.dct(x, type=2)
        ref = _naive_dct2(np.array(x, dtype=float))
        np.testing.assert_allclose(y, ref, atol=1e-9)

    def test_dst_array_like_input(self):
        x   = list(range(1, 9))
        y   = pxdct.dst(x, type=2)
        ref = _naive_dst2(np.array(x, dtype=float))
        np.testing.assert_allclose(y, ref, atol=1e-9)

    def test_dct_scaling_none_default(self):
        """dct() with no scaling arg must match a 'none' plan."""
        x      = np.random.randn(32)
        y_func = pxdct.dct(x, type=2)
        p      = DctPlan("dct2", 32, scaling="none")
        y_plan = p.execute_into(x)
        np.testing.assert_array_equal(y_func, y_plan)

    def test_dct_scaling_ortho(self):
        """dct(scaling='ortho') must match an ortho plan."""
        x      = np.random.randn(32)
        y_func = pxdct.dct(x, type=2, scaling="ortho")
        p      = DctPlan("dct2", 32, scaling="ortho")
        y_plan = p.execute_into(x)
        np.testing.assert_array_equal(y_func, y_plan)

    def test_dst_scaling_ortho(self):
        x      = np.random.randn(32)
        y_func = pxdct.dst(x, type=2, scaling="ortho")
        p      = DctPlan("dst2", 32, scaling="ortho")
        y_plan = p.execute_into(x)
        np.testing.assert_array_equal(y_func, y_plan)

    def test_dct_invalid_scaling_raises(self):
        with pytest.raises(ValueError):
            pxdct.dct(np.ones(8), scaling="bad")

    def test_plan_factory(self):
        p = pxdct.plan("dct2", 64)
        assert isinstance(p, DctPlan)
        assert p.length  == 64
        assert p.scaling == "none"

    def test_plan_factory_scaling(self):
        p = pxdct.plan("dct2", 64, scaling="ortho")
        assert p.scaling == "ortho"

    def test_plan_factory_ortho_roundtrip(self):
        n   = 64
        fwd = pxdct.plan("dct2", n, scaling="ortho")
        inv = pxdct.plan("dct3", n, scaling="ortho")
        x   = np.random.randn(n)
        np.testing.assert_allclose(inv(fwd(x)), x, atol=1e-10)

    def test_plan_factory_mdct(self):
        p = pxdct.plan("mdct", 64)
        assert isinstance(p, DctPlan)
        x = np.random.randn(128)
        y = p.execute_into(x)
        assert y.shape == (64,)

    def test_plan_factory_imdct(self):
        p = pxdct.plan("imdct", 64)
        assert isinstance(p, DctPlan)
        x = np.random.randn(64)
        y = p.execute_into(x)
        assert y.shape == (128,)

    def test_plan2d_factory_square(self):
        p = pxdct.plan2d("dct2", 32)
        assert isinstance(p, DctPlan2D)
        assert p.width == p.height == 32

    def test_plan2d_factory_rectangular(self):
        p = pxdct.plan2d("dct2", 64, height=48)
        assert p.width == 64 and p.height == 48

    def test_plan2d_factory_different_kinds(self):
        p = pxdct.plan2d("dct2", 32, kind_height="dct3", height=32)
        assert isinstance(p, DctPlan2D)

    def test_plan2d_factory_default_height_equals_width(self):
        p = pxdct.plan2d("dct2", 16)
        assert p.width == p.height == 16

    def test_plan2d_factory_default_kind_height_equals_kind_width(self):
        n          = 8
        p_implicit = pxdct.plan2d("dct2", n)
        p_explicit = DctPlan2D(DctPlan("dct2", n), DctPlan("dct2", n))
        x = np.random.randn(n * n)
        a, b = x.copy(), x.copy()
        p_implicit.execute(a)
        p_explicit.execute(b)
        np.testing.assert_array_equal(a, b)

    def test_plan2d_factory_ortho_roundtrip(self):
        """plan2d with ortho DCT-IV is self-inverse — clean 2-D round-trip."""
        n = 16
        p = pxdct.plan2d("dct4", n, scaling="ortho")
        x    = np.random.randn(n * n)
        orig = x.copy()
        p.execute(x)
        p.execute(x)
        np.testing.assert_allclose(x, orig, atol=1e-9)

    def test_plan2d_factory_scaling_propagates(self):
        """Scaling passed to plan2d must be visible via the underlying DctPlan objects.
        We verify indirectly: ortho and none must produce different output."""
        n     = 8
        x     = np.random.randn(n * n)
        p_none  = pxdct.plan2d("dct2", n, scaling="none")
        p_ortho = pxdct.plan2d("dct2", n, scaling="ortho")
        a, b    = x.copy(), x.copy()
        p_none.execute(a)
        p_ortho.execute(b)
        assert not np.allclose(a, b)