//! Dependency relation encoders.

use serde_derive::{Deserialize, Serialize};

mod error;
pub use self::error::*;

mod post_processing;
pub(crate) use self::post_processing::*;

mod relative_position;
pub use self::relative_position::*;

mod relative_pos;
pub use self::relative_pos::*;

/// Encoding of a dependency relation as a token label.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DependencyEncoding<H> {
    head: H,
    label: String,
}

impl<H> DependencyEncoding<H> {
    pub fn new(head: H, label: impl Into<String>) -> Self {
        DependencyEncoding {
            head,
            label: label.into(),
        }
    }

    /// Get the head representation.
    pub fn head(&self) -> &H {
        &self.head
    }

    /// Get the dependency label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use conllu::graph::{Node, Sentence};
    use conllu::io::Reader;

    use super::{POSLayer, RelativePOSEncoder, RelativePositionEncoder};
    use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

    const NON_PROJECTIVE_DATA: &str = "testdata/lassy-small-dev.conllu";

    const ROOT_RELATION: &str = "root";

    fn copy_sentence_without_deprels(sentence: &Sentence) -> Sentence {
        let mut copy = Sentence::new();

        copy.set_comments(sentence.comments().to_owned());

        for token in sentence.iter().filter_map(Node::token) {
            copy.push(token.clone());
        }

        copy
    }

    fn test_encoding<P, E, C>(path: P, encoder_decoder: E)
    where
        P: AsRef<Path>,
        E: SentenceEncoder<Encoding = C> + SentenceDecoder<Encoding = C>,
        C: 'static + Clone,
    {
        let f = File::open(path).unwrap();
        let reader = Reader::new(BufReader::new(f));

        for sentence in reader {
            let sentence = sentence.unwrap();

            // Encode
            let encodings = encoder_decoder
                .encode(&sentence)
                .unwrap()
                .into_iter()
                .map(|e| [EncodingProb::new(e, 1.)])
                .collect::<Vec<_>>();

            // Decode
            let mut test_sentence = copy_sentence_without_deprels(&sentence);
            encoder_decoder
                .decode(&encodings, &mut test_sentence)
                .unwrap();

            assert_eq!(sentence, test_sentence);
        }
    }

    #[test]
    fn relative_pos_position() {
        let encoder = RelativePOSEncoder::new(POSLayer::XPos, ROOT_RELATION);
        test_encoding(NON_PROJECTIVE_DATA, encoder);
    }

    #[test]
    fn relative_position() {
        let encoder = RelativePositionEncoder::new(ROOT_RELATION);
        test_encoding(NON_PROJECTIVE_DATA, encoder);
    }
}
