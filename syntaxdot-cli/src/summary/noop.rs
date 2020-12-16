use anyhow::Result;

use crate::summary::ScalarWriter;

pub struct NoopWriter;

impl ScalarWriter for NoopWriter {
    fn write_scalar(&self, _tag: &str, _step: i64, _value: f32) -> Result<()> {
        Ok(())
    }
}
