use std::io::{self, Write};

use prost::Message;

use crate::event_writer::event::What;
use crate::record_writer::TFRecordWriter;
use std::time::SystemTime;

pub struct EventWriter<W> {
    writer: TFRecordWriter<W>,
}

impl<W> EventWriter<W> {
    fn wall_time() -> f64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f64
            / 1e9
    }
}

impl<W> EventWriter<W>
where
    W: Write,
{
    pub fn new(write: W) -> io::Result<Self> {
        Self::new_with_wall_time(write, Self::wall_time())
    }

    pub fn new_with_wall_time(write: W, wall_time: f64) -> io::Result<Self> {
        let mut writer = EventWriter {
            writer: TFRecordWriter::from(write),
        };

        writer.write_event_with_wall_time(
            wall_time,
            0,
            What::FileVersion("brain.Event:2".to_string()),
        )?;

        Ok(writer)
    }

    pub fn write_event(&mut self, step: i64, what: What) -> io::Result<()> {
        self.write_event_with_wall_time(Self::wall_time(), step, what)
    }

    pub fn write_event_with_wall_time(
        &mut self,
        wall_time: f64,
        step: i64,
        what: What,
    ) -> io::Result<()> {
        let event = Event {
            wall_time,
            step,
            what: Some(what),
        };

        let mut event_bytes = vec![];
        event.encode(&mut event_bytes)?;

        self.writer.write(&event_bytes)?;

        self.writer.flush()
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct Event {
    #[prost(double, tag = "1")]
    wall_time: f64,

    #[prost(int64, tag = "2")]
    step: i64,

    #[prost(oneof = "event::What", tags = "3, 4, 5, 6, 7, 8, 9")]
    what: Option<event::What>,
}

pub mod event {
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum What {
        #[prost(string, tag = "3")]
        FileVersion(std::string::String),

        #[prost(message, tag = "5")]
        Summary(super::Summary),
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct Summary {
    #[prost(message, repeated, tag = "1")]
    pub value: ::std::vec::Vec<summary::Value>,
}

pub mod summary {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct Value {
        #[prost(string, tag = "7")]
        pub node_name: std::string::String,

        #[prost(string, tag = "1")]
        pub tag: std::string::String,

        #[prost(oneof = "value::Value", tags = "2, 3, 4, 5, 6, 8")]
        pub value: ::std::option::Option<value::Value>,
    }

    pub mod value {
        use prost::Oneof;

        #[derive(Clone, PartialEq, Oneof)]
        pub enum Value {
            #[prost(float, tag = "2")]
            SimpleValue(f32),
        }
    }
}
