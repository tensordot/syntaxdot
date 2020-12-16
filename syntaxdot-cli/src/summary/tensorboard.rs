use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;
use syntaxdot_summary::SummaryWriter;

use super::ScalarWriter;

pub struct TensorBoardWriter {
    writer: RefCell<SummaryWriter<BufWriter<File>>>,
}

impl TensorBoardWriter {
    pub fn new(prefix: impl AsRef<str>) -> Result<Self> {
        let writer = SummaryWriter::from_prefix(prefix.as_ref())?;
        Ok(TensorBoardWriter {
            writer: RefCell::new(writer),
        })
    }
}

impl ScalarWriter for TensorBoardWriter {
    fn write_scalar(&self, tag: &str, step: i64, value: f32) -> Result<()> {
        Ok(self.writer.borrow_mut().write_scalar(tag, step, value)?)
    }
}
