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

use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const CASTAGNOLI_POLY: u32 = 0x82f63b78;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(err) = try_main() {
        panic!("{}", err);
    }
}

fn try_main() -> Result<()> {
    let out_dir = match env::var_os("OUT_DIR") {
        None => return Err(From::from("OUT_DIR environment variable not defined")),
        Some(out_dir) => PathBuf::from(out_dir),
    };
    write_tag_lookup_table(&out_dir)?;
    write_crc_tables(&out_dir)?;
    Ok(())
}

fn write_tag_lookup_table(out_dir: &Path) -> Result<()> {
    let out_path = out_dir.join("tag.rs");
    let mut out = io::BufWriter::new(File::create(out_path)?);

    writeln!(out, "pub const TAG_LOOKUP_TABLE: [u16; 256] = [")?;
    for b in 0u8..=255 {
        writeln!(out, "    {},", tag_entry(b))?;
    }
    writeln!(out, "];")?;
    Ok(())
}

fn tag_entry(b: u8) -> u16 {
    let b = b as u16;
    match b & 0b00000011 {
        0b00 => {
            let lit_len = (b >> 2) + 1;
            if lit_len <= 60 {
                lit_len
            } else {
                assert!(lit_len <= 64);
                (lit_len - 60) << 11
            }
        }
        0b01 => {
            let len = 4 + ((b >> 2) & 0b111);
            let offset = (b >> 5) & 0b111;
            (1 << 11) | (offset << 8) | len
        }
        0b10 => {
            let len = 1 + (b >> 2);
            (2 << 11) | len
        }
        0b11 => {
            let len = 1 + (b >> 2);
            (4 << 11) | len
        }
        _ => unreachable!(),
    }
}

fn write_crc_tables(out_dir: &Path) -> Result<()> {
    let out_path = out_dir.join("crc32_table.rs");
    let mut out = io::BufWriter::new(File::create(out_path)?);

    let table = make_table(CASTAGNOLI_POLY);
    let table16 = make_table16(CASTAGNOLI_POLY);

    writeln!(out, "pub const TABLE: [u32; 256] = [")?;
    for &x in table.iter() {
        writeln!(out, "    {},", x)?;
    }
    writeln!(out, "];\n")?;

    writeln!(out, "pub const TABLE16: [[u32; 256]; 16] = [")?;
    for table in table16.iter() {
        writeln!(out, "    [")?;
        for &x in table.iter() {
            writeln!(out, "        {},", x)?;
        }
        writeln!(out, "    ],")?;
    }
    writeln!(out, "];")?;

    out.flush()?;

    Ok(())
}

fn make_table16(poly: u32) -> [[u32; 256]; 16] {
    let mut tab = [[0; 256]; 16];
    tab[0] = make_table(poly);
    for i in 0..256 {
        let mut crc = tab[0][i];
        for j in 1..16 {
            crc = (crc >> 8) ^ tab[0][crc as u8 as usize];
            tab[j][i] = crc;
        }
    }
    tab
}

fn make_table(poly: u32) -> [u32; 256] {
    let mut tab = [0; 256];
    for i in 0u32..256u32 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
        }
        tab[i as usize] = crc;
    }
    tab
}
