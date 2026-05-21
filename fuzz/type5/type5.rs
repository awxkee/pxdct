#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pxdct::Pxdct;

#[derive(Arbitrary, Debug)]
struct DctArbitrary {
    size: u8,
}

fuzz_target!(|data: DctArbitrary| {
    if data.size == 0 {
        return;
    }

    let len = data.size as usize;

    // current workaround for stdarch bug
    if data.size < 30 {
        return;
    }

    let mut array = vec![0.0; len];
    for (i, k) in array.iter_mut().enumerate() {
        *k = i as f32 / len as f32;
    }

    let dct1 = Pxdct::make_dst5_f32(len).unwrap();
    let dst1 = Pxdct::make_dst5_f32(len).unwrap();

    dct1.execute(&mut array).unwrap();
    dst1.execute(&mut array).unwrap();

    let mut array = vec![0.0; len];
    for (i, k) in array.iter_mut().enumerate() {
        *k = i as f64 / len as f64;
    }

    let dct1 = Pxdct::make_dst5_f64(len).unwrap();
    let dst1 = Pxdct::make_dst5_f64(len).unwrap();

    dct1.execute(&mut array).unwrap();
    dst1.execute(&mut array).unwrap();
});
