use conllu::graph::Sentence;
use ndarray::{Array1, ArrayView1};

mod albert;
pub use albert::AlbertTokenizer;

mod bert;
pub use bert::BertTokenizer;

mod xlm_roberta;
use std::ops::{Deref, DerefMut};
pub use xlm_roberta::XlmRobertaTokenizer;

/// Trait for wordpiece tokenizers.
pub trait Tokenize: Send + Sync {
    /// Tokenize a sentence into word pieces.
    #[doc(hidden)]
    fn tokenize_(&self, sentence: &Sentence) -> pieces::PiecesWithOffsets;

    /// Tokenize the tokens in a sentence into word pieces.
    ///
    /// This method takes ownership of the sentence. Use `tokenize_mut` to
    /// store a reference to the sentence.
    fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces<'static> {
        let pieces_offsets = self.tokenize_(&sentence);
        SentenceWithPieces {
            pieces: pieces_offsets.pieces,
            sentence: OwnedOrBorrowed::Owned(sentence),
            token_offsets: pieces_offsets.token_offsets,
        }
    }

    /// Tokenize the tokens in a sentence into word pieces.
    ///
    /// This method takes modified a sentence in-place.
    fn tokenize_mut<'a>(&self, sentence: &'a mut Sentence) -> SentenceWithPieces<'a> {
        let pieces_offsets = self.tokenize_(sentence);
        SentenceWithPieces {
            pieces: pieces_offsets.pieces,
            sentence: OwnedOrBorrowed::Borrowed(sentence),
            token_offsets: pieces_offsets.token_offsets,
        }
    }
}

/// A sentence and its word pieces.
pub struct SentenceWithPieces<'a> {
    /// Word pieces in a sentence.
    pieces: Array1<i64>,

    /// Sentence graph.
    sentence: OwnedOrBorrowed<'a, Sentence>,

    /// The the offsets of tokens in `pieces`.
    token_offsets: Vec<usize>,
}

impl<'a> SentenceWithPieces<'a> {
    /// Get the word piece indices.
    pub fn pieces(&self) -> ArrayView1<i64> {
        self.pieces.view()
    }

    /// Get the sentence graph.
    pub fn sentence(&self) -> &Sentence {
        &self.sentence
    }

    pub fn sentence_mut(&mut self) -> &mut Sentence {
        &mut self.sentence
    }

    /// Get the token offsets.
    pub fn token_offsets(&self) -> &[usize] {
        &self.token_offsets
    }

    /// Decompose the data structure into its parts.
    ///
    /// Returns the sentence, the word piece indices, and the token offsets.
    pub fn into_parts(self) -> (Sentence, Array1<i64>, Vec<usize>) {
        (
            match self.sentence {
                OwnedOrBorrowed::Owned(sent) => sent,
                OwnedOrBorrowed::Borrowed(sent) => sent.clone(),
            },
            self.pieces,
            self.token_offsets,
        )
    }
}

/// Owned or borrowed data.
///
/// We can't use `Cow`, since it does not support mutable references.
enum OwnedOrBorrowed<'a, T: 'a> {
    Owned(T),
    Borrowed(&'a mut T),
}

impl<'a, T> Deref for OwnedOrBorrowed<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            OwnedOrBorrowed::Owned(v) => &v,
            OwnedOrBorrowed::Borrowed(v) => *v,
        }
    }
}

impl<'a, T> DerefMut for OwnedOrBorrowed<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            OwnedOrBorrowed::Owned(ref mut v) => v,
            OwnedOrBorrowed::Borrowed(v) => *v,
        }
    }
}

pub(crate) mod pieces {
    use ndarray::Array1;

    pub struct PiecesWithOffsets {
        pub pieces: Array1<i64>,
        pub token_offsets: Vec<usize>,
    }
}
