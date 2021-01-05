use std::fs::{create_dir_all, File};
use std::io::{self, BufWriter, ErrorKind, Write};
use std::path::PathBuf;

use crate::event_writer::event::What;
use crate::event_writer::summary::value::Value::SimpleValue;
use crate::event_writer::summary::Value;
use crate::event_writer::{EventWriter, Summary};
use std::time::{SystemTime, UNIX_EPOCH};

/// TensorBoard summary writer.
pub struct SummaryWriter<W> {
    writer: EventWriter<W>,
}

impl SummaryWriter<BufWriter<File>> {
    /// Construct a writer from a path prefix.
    ///
    /// For instance, a path such as `tensorboard/bert/opt` will create
    /// the directory `tensorboard/bert` if it does not exist. Within that
    /// directory, it will write to the file
    /// `opt.out.tfevents.<timestamp>.<hostname>`.
    pub fn from_prefix(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();

        if path.components().count() == 0 {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                "summary prefix must not be empty".to_string(),
            ));
        }

        if let Some(dir) = path.parent() {
            create_dir_all(dir)?;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros();
        let hostname = hostname::get()?;

        let mut path_string = path.into_os_string();
        path_string.push(format!(".out.tfevents.{}.", timestamp));
        path_string.push(hostname);

        SummaryWriter::new(BufWriter::new(File::create(path_string)?))
    }
}

impl<W> SummaryWriter<W>
where
    W: Write,
{
    /// Construct a writer from a `Write` type.
    pub fn new(write: W) -> io::Result<Self> {
        let writer = EventWriter::new(write)?;
        Ok(SummaryWriter { writer })
    }

    /// Create a writer that uses the given wall time in the version record.
    ///
    /// This constructor is provided for unit tests.
    #[allow(dead_code)]
    fn new_with_wall_time(write: W, wall_time: f64) -> io::Result<Self> {
        let writer = EventWriter::new_with_wall_time(write, wall_time)?;
        Ok(SummaryWriter { writer })
    }

    /// Write a scalar.
    pub fn write_scalar(
        &mut self,
        tag: impl Into<String>,
        step: i64,
        scalar: f32,
    ) -> std::io::Result<()> {
        self.writer.write_event(
            step,
            What::Summary(Summary {
                value: vec![Value {
                    node_name: "".to_string(),
                    tag: tag.into(),
                    value: Some(SimpleValue(scalar)),
                }],
            }),
        )
    }

    /// Write a scalar with the given wall time.
    ///
    /// This method is provided for unit tests.
    #[allow(dead_code)]
    fn write_scalar_with_wall_time(
        &mut self,
        wall_time: f64,
        tag: impl Into<String>,
        step: i64,
        scalar: f32,
    ) -> std::io::Result<()> {
        self.writer.write_event_with_wall_time(
            wall_time,
            step,
            What::Summary(Summary {
                value: vec![Value {
                    node_name: "".to_string(),
                    tag: tag.into(),
                    value: Some(SimpleValue(scalar)),
                }],
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::SummaryWriter;

    static CHECK_OUTPUT: [u8; 126] = [
        24, 0, 0, 0, 0, 0, 0, 0, 163, 127, 75, 34, 9, 0, 0, 128, 54, 111, 246, 215, 65, 26, 13, 98,
        114, 97, 105, 110, 46, 69, 118, 101, 110, 116, 58, 50, 136, 162, 101, 134, 27, 0, 0, 0, 0,
        0, 0, 0, 26, 13, 158, 19, 9, 188, 119, 164, 54, 111, 246, 215, 65, 16, 10, 42, 14, 10, 12,
        10, 5, 104, 101, 108, 108, 111, 21, 0, 0, 40, 66, 93, 240, 111, 128, 27, 0, 0, 0, 0, 0, 0,
        0, 26, 13, 158, 19, 9, 48, 127, 164, 54, 111, 246, 215, 65, 16, 20, 42, 14, 10, 12, 10, 5,
        119, 111, 114, 108, 100, 21, 0, 0, 128, 63, 5, 210, 83, 151,
    ];

    #[test]
    fn writes_the_same_output_as_tensorflow() {
        let mut data = vec![];
        let mut writer = SummaryWriter::new_with_wall_time(&mut data, 1608105178.).unwrap();
        writer
            .write_scalar_with_wall_time(1608105178.569808, "hello", 10, 42.)
            .unwrap();
        writer
            .write_scalar_with_wall_time(1608105178.570263, "world", 20, 1.)
            .unwrap();

        assert_eq!(data, CHECK_OUTPUT);
    }
}
