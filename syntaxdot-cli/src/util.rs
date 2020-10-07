use std::io::BufRead;
use std::os::raw::c_int;

use anyhow::Result;

pub fn count_conllu_sentences(buf_read: impl BufRead) -> Result<usize> {
    let mut n_sents = 0;

    for line in buf_read.lines() {
        let line = line?;
        if line.starts_with("1\t") {
            n_sents += 1;
        }
    }

    Ok(n_sents)
}

#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}

/// Runs a closure with autocast.
///
/// This function runs a closure with `autocast` if enabled
/// is set to `true`. Otherwise, the closure is run without
/// autocast *iff* the calling function is **not** autocast.
///
/// This function can be used to avoid autocasting overhead.
pub fn autocast_or_preserve<F, T>(enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    if enabled {
        tch::autocast(true, f)
    } else {
        f()
    }
}
