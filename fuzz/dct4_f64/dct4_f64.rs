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
        *k = i as f64 / len as f64;
    }

    let mut zap = vec![0.0; len];

    let dct4 = Pxdct::make_dct4_f64(len).unwrap();

    dct4.execute(&mut array).unwrap();
    dct4.execute_into(&mut array, &mut zap).unwrap();
});
