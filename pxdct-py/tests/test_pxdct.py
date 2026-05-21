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


SIZES = [1, 2, 4, 7, 8, 12, 16, 32, 100, 128, 256]

# ─── DctPlan construction ────────────────────────────────────────────────────

class TestDctPlanConstruction:
    def test_basic_attributes(self):
        p = DctPlan("dct2", 64)
        assert p.length == 64
        assert p.kind == "dct2"
        assert p.dtype == "f64"

    def test_f32_dtype(self):
        p = DctPlan("dct2", 64, "f32")
        assert p.dtype == "f32"

    def test_repr(self):
        p = DctPlan("dst4", 32, "f64")
        assert "dst4" in repr(p)
        assert "32" in repr(p)

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

# ─── execute (in-place) ──────────────────────────────────────────────────────

class TestExecuteInPlace:
    @pytest.mark.parametrize("n", SIZES)
    def test_dct2_matches_naive(self, n):
        x = np.random.randn(n)
        ref = _naive_dct2(x)
        p = DctPlan("dct2", n)
        data = x.copy()
        p.execute(data)
        np.testing.assert_allclose(data, ref, atol=1e-9)

    @pytest.mark.parametrize("n", SIZES)
    def test_roundtrip_dct2_dct3(self, n):
        """DCT-II followed by DCT-III (scaled) should recover the input."""
        x = np.random.randn(n)
        fwd = DctPlan("dct2", n)
        inv = DctPlan("dct3", n)
        y = x.copy()
        fwd.execute(y)
        inv.execute(y)
        y *= 2.0 / n
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_wrong_length_raises(self):
        p = DctPlan("dct2", 16)
        with pytest.raises(ValueError, match="length"):
            p.execute(np.ones(8))

    def test_wrong_dtype_raises(self):
        p = DctPlan("dct2", 8)           # f64 plan
        with pytest.raises(ValueError):
            p.execute(np.ones(8, dtype=np.float32))

    def test_f32_plan(self):
        n = 32
        p = DctPlan("dct2", n, "f32")
        x = np.random.randn(n).astype(np.float32)
        ref = _naive_dct2(x.astype(float)).astype(np.float32)
        p.execute(x)
        np.testing.assert_allclose(x, ref, atol=1e-4)   # f32 tolerance


# ─── execute_into ─────────────────────────────────────────────────────────────

class TestExecuteInto:
    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_does_not_modify_input(self, n):
        x = np.random.randn(n)
        original = x.copy()
        out = np.empty(n)
        p = DctPlan("dct2", n)
        p.execute_into(x, out)
        np.testing.assert_array_equal(x, original)

    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_output_matches_inplace(self, n):
        x = np.random.randn(n)
        p = DctPlan("dct2", n)
        inplace = x.copy(); p.execute(inplace)
        out = np.empty(n)
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
        n = 16
        p = DctPlan(kind, n)
        x = np.random.randn(n).copy()
        p.execute(x)              # must not raise
        assert np.all(np.isfinite(x))

# ─── DctPlan2D ───────────────────────────────────────────────────────────────

class TestDctPlan2D:
    def test_attributes(self):
        wp = DctPlan("dct2", 32)
        hp = DctPlan("dct2", 64)
        p2 = DctPlan2D(wp, hp)
        assert p2.width == 32
        assert p2.height == 64

    def test_dtype_mismatch_raises(self):
        wp = DctPlan("dct2", 16, "f32")
        hp = DctPlan("dct2", 16, "f64")
        with pytest.raises(ValueError):
            DctPlan2D(wp, hp)

    def test_square_roundtrip(self):
        n = 16
        wp = DctPlan("dct2", n)
        hp = DctPlan("dct2", n)
        p2f = DctPlan2D(wp, hp)

        wi = DctPlan("dct3", n)
        hi = DctPlan("dct3", n)
        p2i = DctPlan2D(wi, hi)

        x = np.random.randn(n * n)
        orig = x.copy()
        p2f.execute(x)
        p2i.execute(x)
        x *= (2.0 / n) ** 2
        np.testing.assert_allclose(x, orig, atol=1e-9)

    def test_wrong_flat_size_raises(self):
        wp = DctPlan("dct2", 8)
        hp = DctPlan("dct2", 8)
        p2 = DctPlan2D(wp, hp)
        with pytest.raises(ValueError):
            p2.execute(np.ones(32))   # 32 ≠ 8×8 = 64

# ─── convenience API ─────────────────────────────────────────────────────────

class TestConvenienceAPI:
    def test_dct_function(self):
        x = np.random.randn(32)
        ref = DctPlan("dct2", 32)
        y_ref = x.copy(); ref.execute(y_ref)
        y = pxdct.dct(x, type=2)
        np.testing.assert_allclose(y, y_ref, atol=1e-12)

    def test_dst_function(self):
        x = np.random.randn(32)
        y = pxdct.dst(x, type=2)
        ref = _naive_dst2(x)
        np.testing.assert_allclose(y, ref, atol=1e-9)

    def test_dct_kind_kwarg(self):
        x = np.random.randn(16)
        y1 = pxdct.dct(x, type=4)
        y2 = pxdct.dct(x, kind="dct4")
        np.testing.assert_array_equal(y1, y2)

    def test_plan_factory(self):
        p = pxdct.plan("dct2", 64)
        assert isinstance(p, DctPlan)
        assert p.length == 64

    def test_plan2d_factory_square(self):
        p = pxdct.plan2d("dct2", 32)
        assert isinstance(p, DctPlan2D)
        assert p.width == 32
        assert p.height == 32

    def test_plan2d_factory_rectangular(self):
        p = pxdct.plan2d("dct2", 64, height=48)
        assert p.width == 64
        assert p.height == 48

    def test_plan2d_factory_different_kinds(self):
        p = pxdct.plan2d("dct2", 32, kind_height="dct3", height=32)
        assert isinstance(p, DctPlan2D)