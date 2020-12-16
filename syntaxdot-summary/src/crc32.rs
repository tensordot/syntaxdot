// Copyright 2011, The Snappy-Rust Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use std::convert::TryInto;

use crate::crc32_table::{TABLE, TABLE16};

/// Provides a simple API to generate "masked" CRC32C checksums specifically
/// for use in Snappy. When available, this will make use of SSE 4.2 to compute
/// checksums. Otherwise, it falls back to only-marginally-slower "slicing by
/// 16" technique.
///
/// The main purpose of this type is to cache the CPU feature check and expose
/// a safe API.
#[derive(Clone, Copy, Debug)]
pub struct CheckSummer {
    sse42: bool,
}

impl CheckSummer {
    /// Create a new checksummer that can compute CRC32C checksums on arbitrary
    /// bytes.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn new() -> CheckSummer {
        CheckSummer { sse42: false }
    }

    /// Create a new checksummer that can compute CRC32C checksums on arbitrary
    /// bytes.
    #[cfg(target_arch = "x86_64")]
    pub fn new() -> CheckSummer {
        CheckSummer {
            sse42: is_x86_feature_detected!("sse4.2"),
        }
    }

    /// Returns the "masked" CRC32 checksum of `buf` using the Castagnoli
    /// polynomial. This "masked" checksum is defined by the Snappy frame
    /// format. Masking is supposed to make the checksum robust with respect to
    /// the data that contains the checksum itself.
    pub fn crc32c_masked(&self, buf: &[u8]) -> u32 {
        let sum = self.crc32c(buf);
        (sum.wrapping_shr(15) | sum.wrapping_shl(17)).wrapping_add(0xA282EAD8)
    }

    /// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
    #[cfg(not(target_arch = "x86_64"))]
    fn crc32c(&self, buf: &[u8]) -> u32 {
        crc32c_slice16(buf)
    }

    /// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
    #[cfg(target_arch = "x86_64")]
    fn crc32c(&self, buf: &[u8]) -> u32 {
        if self.sse42 {
            // SAFETY: When sse42 is true, we are guaranteed to be running on
            // a CPU that supports SSE 4.2.
            unsafe { crc32c_sse(buf) }
        } else {
            crc32c_slice16(buf)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse(buf: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let mut crc = !0u32;
    // SAFETY: This is safe since alignment is handled by align_to (oh how I
    // love you) and since 8 adjacent u8's are guaranteed to have the same
    // in-memory representation as u64 for all possible values.
    let (prefix, u64s, suffix) = buf.align_to::<u64>();
    for &b in prefix {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u8(crc, b);
    }
    for &n in u64s {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u64(crc as u64, n) as u32;
    }
    for &b in suffix {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u8(crc, b);
    }
    !crc
}

/// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
fn crc32c_slice16(mut buf: &[u8]) -> u32 {
    let mut crc: u32 = !0;
    while buf.len() >= 16 {
        crc ^= read_u32_le(buf);
        crc = TABLE16[0][buf[15] as usize]
            ^ TABLE16[1][buf[14] as usize]
            ^ TABLE16[2][buf[13] as usize]
            ^ TABLE16[3][buf[12] as usize]
            ^ TABLE16[4][buf[11] as usize]
            ^ TABLE16[5][buf[10] as usize]
            ^ TABLE16[6][buf[9] as usize]
            ^ TABLE16[7][buf[8] as usize]
            ^ TABLE16[8][buf[7] as usize]
            ^ TABLE16[9][buf[6] as usize]
            ^ TABLE16[10][buf[5] as usize]
            ^ TABLE16[11][buf[4] as usize]
            ^ TABLE16[12][(crc >> 24) as u8 as usize]
            ^ TABLE16[13][(crc >> 16) as u8 as usize]
            ^ TABLE16[14][(crc >> 8) as u8 as usize]
            ^ TABLE16[15][(crc) as u8 as usize];
        buf = &buf[16..];
    }
    for &b in buf {
        crc = TABLE[((crc as u8) ^ b) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Read a u32 in little endian format from the beginning of the given slice.
/// This panics if the slice has length less than 4.
pub fn read_u32_le(slice: &[u8]) -> u32 {
    u32::from_le_bytes(slice[..4].try_into().unwrap())
}
