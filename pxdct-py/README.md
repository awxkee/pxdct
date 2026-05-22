# pxdct — Fast DCT/DST for Python

Rust-backed DCT and DST types I–VIII with SIMD, exposed to
Python via [maturin](https://github.com/PyO3/maturin) and
[PyO3](https://pyo3.rs).

`scipy.fft` covers DCT types I–IV only. `pxdct` adds types V–VIII for both
cosine and sine families, MDCT/IMDCT, pre-planned reusable executors, a 2-D
executor for image processing, and three normalization modes including
orthonormal scaling.

## Install

```bash
pip install pxdct
```

Build from source (requires a Rust toolchain):

```bash
pip install maturin
maturin develop --release  # installs into the current venv
```

## Quick start

```python
import numpy as np
import pxdct

x = np.random.randn(256)

# one-shot
y = pxdct.dct(x, type=2)                   # DCT-II, unscaled
y = pxdct.dct(x, type=4)                   # DCT-IV
y = pxdct.dst(x, type=7)                   # DST-VII
y = pxdct.dct(x, type=2, scaling='ortho')  # orthonormal DCT-II

# reusable plan
plan = pxdct.plan('dct2', 256)
out  = np.empty(256)

for frame in audio_frames:
    plan.execute_into(frame, out)

# in-place variant
plan.execute(out)   # overwrites out

# ortho-normalised round-trip (no manual rescaling needed)
fwd = pxdct.plan('dct2', 256, scaling='ortho')
inv = pxdct.plan('dct3', 256, scaling='ortho')
y      = fwd(x)
x_back = inv(y)     # ≈ x

# MDCT / IMDCT (length must be even; scaling is ignored)
mdct_plan  = pxdct.plan('mdct',  256)
imdct_plan = pxdct.plan('imdct', 256)
coeffs = mdct_plan(x)
x_back = imdct_plan(coeffs)

# 2-D
p2 = pxdct.plan2d('dct2', 512)              # 512×512, same kind on both axes
p2 = pxdct.plan2d('dct2', 640, height=480)  # rectangular

img_flat = image.ravel().astype('float64')   # row-major, length = width × height
p2.execute(img_flat)

# 2-D ortho round-trip with DCT-IV (self-inverse under ortho)
p = pxdct.plan2d('dct4', 8, scaling='ortho') 
p.execute(block)   # forward
p.execute(block)   # inverse — recovers original
```

## API reference

### `pxdct.dct(x, type=2, *, kind=None, dtype='f64', scaling='none') → ndarray`
### `pxdct.dst(x, type=2, *, kind=None, dtype='f64', scaling='none') → ndarray`

One-shot transforms. `type` selects DCT/DST type 1–8. `kind` overrides
`type` with an explicit string such as `"dct4"`, `"dst7"`, `"mdct"`, or
`"imdct"`. `scaling` controls normalization (see [Scaling](#scaling) below).

### `pxdct.plan(kind, length, dtype='f64', scaling='none') → DctPlan`

Factory for `DctPlan`.

### `pxdct.plan2d(kind_width, width, kind_height=None, height=None, dtype='f64', scaling='none') → DctPlan2D`

Factory for `DctPlan2D`. `kind_height` and `height` default to `kind_width`
and `width` (square, same kind on both axes). `scaling` is applied to both
axes.

### `class DctPlan`

| Method / attribute              | Description                                                                  |
|---------------------------------|------------------------------------------------------------------------------|
| `execute(data)`                 | In-place transform on a 1-D numpy array                                      |
| `execute_into(input[, output])` | Out-of-place; `input` is not modified; allocates output if omitted           |
| `plan(input[, output])`         | Alias for `execute_into` via `__call__`                                      |
| `.length`                       | Transform size                                                               |
| `.kind`                         | Kind string, e.g. `"dct2"`                                                   |
| `.dtype`                        | `"f32"` or `"f64"`                                                           |
| `.scaling`                      | Normalization mode: `"none"`, `"scale"`, or `"ortho"`                        |

### `class DctPlan2D`

| Method / attribute            | Description                                                             |
|-------------------------------|-------------------------------------------------------------------------|
| `execute(data)`               | In-place transform on a flat row-major array of length `width × height` |
| `.width`, `.height`, `.dtype` | Read-only attributes                                                    |

## Supported transforms

| Kind          | Full name                        | Inverse of             |
|---------------|----------------------------------|------------------------|
| `dct1`        | DCT type I                       | itself (up to scaling) |
| `type2`        | DCT type II (the "standard" DCT) | `dct3`                 |
| `dct3`        | DCT type III (inverse DCT)       | `type2`                 |
| `type4`        | DCT type IV                      | itself                 |
| `dct5`–`dct8` | DCT types V–VIII                 | see literature         |
| `dst1`–`dst8` | DST types I–VIII                 | see literature         |
| `mdct`        | Modified DCT                     | `imdct`                |
| `imdct`       | Inverse MDCT                     | `mdct`                 |

## Scaling

Three normalization modes are available via the `scaling` parameter:

| `scaling` | Effect                                                                            |
|-----------|-----------------------------------------------------------------------------------|
| `"none"`  | Unscaled output                                                                   |
| `"scale"` | Multiply every output element by `sqrt(2 / N)`                                    |
| `"ortho"` | Per-type orthonormal scaling — a forward/inverse pair round-trips to the identity |

`scaling` is ignored for `"mdct"` and `"imdct"`.

### Unscaled round-trip (default)

```python
y         = pxdct.dct(x, type=2)   # DCT-II
recovered = pxdct.dct(y, type=3)   # DCT-III
recovered *= 2.0 / len(x)          # manual rescaling required
```

### Ortho round-trip (no manual rescaling)

```python
fwd = pxdct.plan('dct2', len(x), scaling='ortho')
inv = pxdct.plan('dct3', len(x), scaling='ortho')
y         = fwd(x)
recovered = inv(y)   # ≈ x, no rescaling needed
```

### 2-D ortho note

The 2-D executor applies the 1-D transform independently along rows and
columns. For a clean 2-D ortho round-trip without manual rescaling, use
**DCT-IV**, which is self-inverse under ortho scaling on every axis:

```python
p = pxdct.plan2d('dct4', 8, scaling='ortho')
p.execute(block)   # forward
p.execute(block)   # inverse — recovers original exactly
```

A DCT-II/III ortho pair also works in 2-D, but requires the inverse plan to
apply DCT-III on both axes independently (not a single combined plan object).

## Precision

| `dtype` | Description                             |
|---------|-----------------------------------------|
| `"f64"` | `float64` — default                     |
| `"f32"` | `float32` — ~2× faster on SIMD hardware |