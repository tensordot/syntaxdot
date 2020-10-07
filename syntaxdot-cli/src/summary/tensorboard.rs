use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;
use tfrecord::{EventWriter, EventWriterInit};

use super::SummaryWriter;

pub struct TensorBoardWriter {
    writer: RefCell<EventWriter<BufWriter<File>>>,
}

impl TensorBoardWriter {
    pub fn new(prefix: impl AsRef<str>) -> Result<Self> {
        let writer = EventWriterInit::default().from_prefix(prefix, None)?;
        Ok(TensorBoardWriter {
            writer: RefCell::new(writer),
        })
    }
}

impl SummaryWriter for TensorBoardWriter {
    fn write_scalar(&self, tag: &str, step: i64, value: f32) -> Result<()> {
        Ok(self.writer.borrow_mut().write_scalar(tag, step, value)?)
    }
}
