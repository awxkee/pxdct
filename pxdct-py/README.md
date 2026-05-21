# pxdct — Fast DCT/DST for Python

Rust-backed DCT and DST types I–VIII with SIMD, exposed to
Python via [maturin](https://github.com/PyO3/maturin) and
[PyO3](https://pyo3.rs).

`scipy.fft` covers DCT types I–IV only. `pxdct` adds types V–VIII for both
cosine and sine families, MDCT/IMDCT, pre-planned reusable executors, and a 2-D executor for image processing.

## Install

```bash
pip install pxdct          # pre-built wheel from PyPI
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
y = pxdct.dct(x, type=2)     # DCT-II
y = pxdct.dct(x, type=4)     # DCT-IV
y = pxdct.dst(x, type=7)     # DST-VII

# reusable plan
plan = pxdct.plan('dct2', 256)
out  = np.empty(256)

for frame in audio_frames:
    plan.execute_into(frame, out)

# in-place variant
plan.execute(out)   # overwrites out

# MDCT / IMDCT (length must be even)
mdct_plan  = pxdct.plan('mdct',  256)
imdct_plan = pxdct.plan('imdct', 256)
coeffs = mdct_plan(x)
x_back = imdct_plan(coeffs)

# 2-D
p2 = pxdct.plan2d('dct2', 512)             # 512×512, same kind on both axes
p2 = pxdct.plan2d('dct2', 640, height=480) # rectangular

img_flat = image.ravel().astype('float64')  # row-major, length = width × height
p2.execute(img_flat)
```

## API reference

### `pxdct.dct(x, type=2, *, kind=None, dtype='f64') → ndarray`
### `pxdct.dst(x, type=2, *, kind=None, dtype='f64') → ndarray`

One-shot transforms.  `type` selects DCT/DST type 1–8.  `kind` overrides
`type` with an explicit string such as `"dct4"`, `"dst7"`, `"mdct"`, or
`"imdct"`.

### `pxdct.plan(kind, length, dtype='f64') → DctPlan`

Factory for :class:`DctPlan`.

### `pxdct.plan2d(kind_width, width, kind_height=None, height=None, dtype='f64') → DctPlan2D`

Factory for :class:`DctPlan2D`.  `kind_height` and `height` default to
`kind_width` and `width` (square, same kind on both axes).

### `class DctPlan`

| Method                        | Description                             |
|-------------------------------|-----------------------------------------|
| `execute(data)`               | In-place transform on a 1-D numpy array |
| `execute_into(input, output)` | Out-of-place; `input` is not modified   |
| `.length`                     | Transform size                          |
| `.kind`                       | Kind string, e.g. `"dct2"`              |
| `.dtype`                      | `"f32"` or `"f64"`                      |

### `class DctPlan2D`

| Method                        | Description                                                             |
|-------------------------------|-------------------------------------------------------------------------|
| `execute(data)`               | In-place transform on a flat row-major array of length `width × height` |
| `.width`, `.height`, `.dtype` | Read-only attributes                                                    |

## Supported transforms

| Kind          | Full name                        | Inverse of                          |
|---------------|----------------------------------|-------------------------------------|
| `dct1`        | DCT type I                       | itself (self-inverse up to scaling) |
| `dct2`        | DCT type II (the "standard" DCT) | `dct3`                              |
| `dct3`        | DCT type III (inverse DCT)       | `dct2`                              |
| `dct4`        | DCT type IV                      | itself                              |
| `dct5`–`dct8` | DCT types V–VIII                 | see literature                      |
| `dst1`–`dst8` | DST types I–VIII                 | see literature                      |
| `mdct`        | Modified DCT                     | `imdct`                             |
| `imdct`       | Inverse MDCT                     | `mdct`                              |

## Scaling

By default `pxdct.dct(x, type=2)` returns the **unscaled** DCT-II (matching
`scipy.fft.dct(x, type=*, norm=None)`).

To recover `x` from a DCT-II result `y`:
```python
recovered = pxdct.dct(y, type=3)
recovered *= 2.0 / len(x)
```

## Precision

| `dtype` | Description                             |
|---------|-----------------------------------------|
| `"f64"` | `float64` — default                     |
| `"f32"` | `float32` — ~2× faster on SIMD hardware |