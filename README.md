# Pxdct — Discrete Cosine and Sine Transform Factory

`Pxdct` is the main entry point for creating optimized **DCT (Discrete Cosine Transform)** and **DST (Discrete Sine
Transform)** executors.  
It provides a unified API for constructing fast, in-place transform executors using either **single (`f32`)** or *
*double (`f64`)** precision.

All executors implement the [`PxdctExecutor`] trait and can be used to perform forward or inverse transforms directly on
a mutable data slice.

---

## Features

- Supports **DCT-II**, **DCT-III**, **DST-II**, and **DST-III**
- Works with both `f32` and `f64`

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