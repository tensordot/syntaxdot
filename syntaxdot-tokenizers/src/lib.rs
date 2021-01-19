use conllu::graph::Sentence;
use ndarray::Array1;

mod albert;
pub use albert::AlbertTokenizer;

mod bert;
pub use bert::BertTokenizer;

mod error;
pub use error::TokenizerError;

mod xlm_roberta;
pub use xlm_roberta::XlmRobertaTokenizer;

/// Trait for wordpiece tokenizers.
pub trait Tokenize: Send + Sync {
    /// Tokenize the tokens in a sentence into word pieces.
    ///
    /// Implementations **must** prefix the first piece corresponding to a
    /// token by one or more special pieces marking the beginning of the
    /// sentence. The representation of this piece can be used for special
    /// purposes, such as classification or acting is the pseudo-root in
    /// dependency parsing.
    fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces;
}

/// A sentence and its word pieces.
pub struct SentenceWithPieces {
    /// Word pieces in a sentence.
    pub pieces: Array1<i64>,

    /// Sentence graph.
    pub sentence: Sentence,

    /// The the offsets of tokens in `pieces`.
    pub token_offsets: Vec<usize>,
}
