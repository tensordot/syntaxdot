use std::path::Path;

use sentencepiece::SentencePieceProcessor;
use udgraph::graph::{Node, Sentence};

use super::{SentenceWithPieces, Tokenize};
use crate::TokenizerError;

const FAIRSEQ_BOS_ID: i64 = 0;
const FAIRSEQ_EOS_ID: i64 = 2;
const FAIRSEQ_OFFSET: i64 = 1;
const FAIRSEQ_UNK: i64 = 3;

/// Tokenizer for Roberta models.
///
/// Roberta uses the sentencepiece tokenizer. However, we cannot use
/// it in the intended way: we would have to detokenize sentences and
/// it is not guaranteed that each token has a unique piece, which is
/// required in sequence labeling. So instead, we use the tokenizer as
/// a subword tokenizer.
pub struct XlmRobertaTokenizer {
    spp: SentencePieceProcessor,
}

impl XlmRobertaTokenizer {
    pub fn new(spp: SentencePieceProcessor) -> Self {
        XlmRobertaTokenizer { spp }
    }

    pub fn open<P>(model: P) -> Result<Self, TokenizerError>
    where
        P: AsRef<Path>,
    {
        let spp = SentencePieceProcessor::open(model)?;
        Ok(Self::new(spp))
    }
}

impl From<SentencePieceProcessor> for XlmRobertaTokenizer {
    fn from(spp: SentencePieceProcessor) -> Self {
        XlmRobertaTokenizer::new(spp)
    }
}

impl Tokenize for XlmRobertaTokenizer {
    fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces {
        // An average of three pieces per token ought to be enough for
        // everyone ;).
        let mut pieces = Vec::with_capacity((sentence.len() - 1) * 3);
        let mut token_offsets = Vec::with_capacity(sentence.len());

        pieces.push(FAIRSEQ_BOS_ID);

        for token in sentence.iter().filter_map(Node::token) {
            token_offsets.push(pieces.len());

            let token_pieces = self
                .spp
                .encode(token.form())
                .expect("The sentencepiece tokenizer failed");

            if !token_pieces.is_empty() {
                pieces.extend(token_pieces.into_iter().map(|piece| {
                    let piece_id = piece.id as i64;
                    if piece_id == self.spp.unk_id() as i64 {
                        FAIRSEQ_UNK
                    } else {
                        piece_id + FAIRSEQ_OFFSET
                    }
                }));
            } else {
                // Use the unknown token id if sentencepiece does not
                // give an output for the token. This should not
                // happen under normal circumstances, since
                // sentencepiece does return this id for unknown
                // tokens. However, the input may be corrupt and use
                // some form of non-tab whitespace as a form, for which
                // sentencepiece does not return any identifier.
                pieces.push(FAIRSEQ_UNK);
            }
        }

        pieces.push(FAIRSEQ_EOS_ID);

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

    use ndarray::array;
    use sentencepiece::SentencePieceProcessor;
    use udgraph::graph::Sentence;
    use udgraph::token::Token;

    use super::XlmRobertaTokenizer;
    use crate::Tokenize;

    fn sentence_from_forms(forms: &[&str]) -> Sentence {
        Sentence::from_iter(forms.iter().map(|&f| Token::new(f)))
    }

    fn xlm_roberta_tokenizer() -> XlmRobertaTokenizer {
        let spp = SentencePieceProcessor::open(env!("XLM_ROBERTA_BASE_SENTENCEPIECE")).unwrap();
        XlmRobertaTokenizer::from(spp)
    }

    #[test]
    fn tokenizer_gives_expected_output() {
        let tokenizer = xlm_roberta_tokenizer();
        let sent = sentence_from_forms(&["Veruntreute", "die", "AWO", "Spendengeld", "?"]);
        let pieces = tokenizer.tokenize(sent);
        assert_eq!(
            pieces.pieces,
            array![0, 310, 23451, 107, 6743, 68, 62, 43789, 207126, 49004, 705, 2]
        );
    }

    #[test]
    fn handles_missing_sentence_pieces() {
        let tokenizer = xlm_roberta_tokenizer();
        let sent = sentence_from_forms(&["die", " ", "AWO"]);
        let pieces = tokenizer.tokenize(sent);
        assert_eq!(pieces.pieces, array![0, 68, 1, 62, 43789, 2]);
    }
}
