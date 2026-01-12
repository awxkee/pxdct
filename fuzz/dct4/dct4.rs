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

    dct2.execute(&mut array).unwrap();
});
