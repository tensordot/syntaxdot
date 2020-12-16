use std::io::{self, Write};

use crate::crc32::CheckSummer;

/// Write data in TFRecord format.
pub struct TFRecordWriter<W> {
    checksummer: CheckSummer,
    write: W,
}

impl<W> From<W> for TFRecordWriter<W>
where
    W: Write,
{
    fn from(write: W) -> Self {
        TFRecordWriter {
            checksummer: CheckSummer::new(),
            write,
        }
    }
}

impl<W> TFRecordWriter<W>
where
    W: Write,
{
    pub fn flush(&mut self) -> io::Result<()> {
        self.write.flush()
    }

    pub fn write(&mut self, data: &[u8]) -> io::Result<()> {
        let len = (data.len() as u64).to_le_bytes();
        self.write.write_all(&len)?;
        self.write
            .write_all(&self.checksummer.crc32c_masked(&len).to_le_bytes())?;
        self.write.write_all(data)?;
        self.write
            .write_all(&self.checksummer.crc32c_masked(data).to_le_bytes())?;

        Ok(())
    }
}

// TFRecord format:
//
// uint64 length
// uint32 masked_crc32_of_length
// byte   data[length]
// uint32 masked_crc32_of_data
