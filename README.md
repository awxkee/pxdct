# Pxdct — Discrete Cosine and Sine Transform Factory

Pxdct is a high-performance DCT/DST library for Rust supporting arbitrary transform lengths, not just powers of 2.

## Why Pxdct

Most DCT libraries require power-of-2 window sizes. Pxdct efficiently handles any length by automatically selecting the best algorithm:

- Split-radix for power-of-2 sizes
- Mixed-radix decomposition for sizes with small prime factors (3, 5, 7, 11, 13…)
- Prime Factor Algorithm (PFA) for coprime factorizations
- Butterflies for sizes up to 512

Sizes that are products of small primes (2, 3, 5, 7, 11, 13) perform best. Large prime sizes fall back to a general FFT-based path and are still O(n log n) but slower than smooth numbers of similar magnitude.

Hardware acceleration is applied automatically where available (AVX2, NEON).

Hardware acceleration is applied automatically:
- AVX2
- NEON

---

## Supported Transforms

| Transform         | Description           |
|-------------------|-----------------------|
| DCT-I / DST-I     | Type 1                |
| DCT-II / DST-II   | The classic DCT / DST |
| DCT-III / DST-III | Inverse of type II    |
| DCT-IV            | Used in MDCT          |
| DCT-VII / DST-VII | Type 7                |

Both `f32` and `f64` precision are supported.

---

## Example

```rust
use pxdct::Pxdct;
use pxdct::PxdctExecutor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Create a DCT-II executor for f32
    let dct2 = Pxdct::make_dct2_f32(data.len())?;
    dct2.execute(&mut data)?;

    println!("Transformed data: {:?}", data);
    Ok(())
}
```

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.