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

    let mut array = vec![0.0; len];
    for (i, k) in array.iter_mut().enumerate() {
        *k = i as f32 / len as f32;
    }

    let dct2 = Pxdct::make_dct2_f32(len).unwrap();
    let dct3 = Pxdct::make_dct3_f32(len).unwrap();

    let dst2 = Pxdct::make_dst2_f32(len).unwrap();
    let dst3 = Pxdct::make_dst3_f32(len).unwrap();

    dct2.execute(&mut array).unwrap();
    dct3.execute(&mut array).unwrap();
    dst2.execute(&mut array).unwrap();
    dst3.execute(&mut array).unwrap();
});
