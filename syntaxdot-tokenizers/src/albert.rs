use std::path::Path;

use conllu::graph::{Node, Sentence};
use sentencepiece::SentencePieceProcessor;

use super::{SentenceWithPieces, Tokenize};
use crate::TokenizerError;

/// Tokenizer for ALBERT models.
///
/// ALBERT uses the sentencepiece tokenizer. However, we cannot use
/// it in the intended way: we would have to detokenize sentences and
/// it is not guaranteed that each token has a unique piece, which is
/// required in sequence labeling. So instead, we use the tokenizer as
/// a subword tokenizer.
pub struct AlbertTokenizer {
    spp: SentencePieceProcessor,
}

impl AlbertTokenizer {
    pub fn new(spp: SentencePieceProcessor) -> Self {
        AlbertTokenizer { spp }
    }

    pub fn open<P>(model: P) -> Result<Self, TokenizerError>
    where
        P: AsRef<Path>,
    {
        let spp = SentencePieceProcessor::load(&model.as_ref().to_string_lossy())?;
        Ok(Self::new(spp))
    }
}

impl From<SentencePieceProcessor> for AlbertTokenizer {
    fn from(spp: SentencePieceProcessor) -> Self {
        AlbertTokenizer::new(spp)
    }
}

impl Tokenize for AlbertTokenizer {
    fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces {
        // An average of three pieces per token ought to be enough for
        // everyone ;).
        let mut pieces = Vec::with_capacity((sentence.len() + 1) * 3);
        let mut token_offsets = Vec::with_capacity(sentence.len());

        pieces.push(
            self.spp
                .piece_to_id("[CLS]")
                .expect("ALBERT model does not have a [CLS] token")
                .expect("ALBERT model does not have a [CLS] token") as i64,
        );

        for token in sentence.iter().filter_map(Node::token) {
            token_offsets.push(pieces.len());

            let token_pieces = self
                .spp
                .encode(token.form())
                .expect("The sentencepiece tokenizer failed");

            if !token_pieces.is_empty() {
                pieces.extend(token_pieces.into_iter().map(|piece| piece.id as i64));
            } else {
                // Use the unknown token id if sentencepiece does not
                // give an output for the token. This should not
                // happen under normal circumstances, since
                // sentencepiece does return this id for unknown
                // tokens. However, the input may be corrupt and use
                // some form of non-tab whitespace as a form, for which
                // sentencepiece does not return any identifier.
                pieces.push(self.spp.unknown_id() as i64);
            }
        }

        pieces.push(
            self.spp
                .piece_to_id("[SEP]")
                .expect("ALBERT model does not have a [SEP] token")
                .expect("ALBERT model does not have a [SEP] token") as i64,
        );

        SentenceWithPieces {
            pieces: pieces.into(),
            sentence,
            token_offsets,
        }
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use conllu::graph::Sentence;
    use conllu::token::Token;
    use ndarray::array;
    use sentencepiece::SentencePieceProcessor;

    use super::AlbertTokenizer;
    use crate::Tokenize;

    fn sentence_from_forms(forms: &[&str]) -> Sentence {
        Sentence::from_iter(forms.iter().map(|&f| Token::new(f)))
    }

    fn albert_tokenizer() -> AlbertTokenizer {
        let spp = SentencePieceProcessor::load(env!("ALBERT_BASE_V2_SENTENCEPIECE")).unwrap();
        AlbertTokenizer::new(spp)
    }

    #[test]
    fn tokenizer_gives_expected_output() {
        let tokenizer = albert_tokenizer();
        let sent = sentence_from_forms(&["pierre", "vinken", "will", "join", "the", "board", "."]);
        let pieces = tokenizer.tokenize(sent);
        assert_eq!(
            pieces.pieces,
            array![2, 5399, 9730, 2853, 129, 1865, 14, 686, 13, 9, 3]
        );
    }

    #[test]
    fn handles_missing_sentence_pieces() {
        let tokenizer = albert_tokenizer();
        let sent = sentence_from_forms(&["pierre", " ", "vinken"]);
        let pieces = tokenizer.tokenize(sent);
        assert_eq!(pieces.pieces, array![2, 5399, 1, 9730, 2853, 3]);
    }
}
