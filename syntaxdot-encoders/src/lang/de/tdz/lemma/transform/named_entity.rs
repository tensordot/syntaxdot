use std::iter;
use std::iter::FromIterator;

use caseless::Caseless;
use seqalign::op::{archetype, Operation};
use seqalign::{Align, Measure, SeqPair};
use unicode_normalization::UnicodeNormalization;

/// Levenshtein distance with case a case-insensitive match operation.
#[derive(Clone, Debug)]
struct CaseInsensitiveLevenshtein {
    ops: [CaseInsensitiveLevenshteinOp; 4],
}

impl CaseInsensitiveLevenshtein {
    /// Construct a Levenshtein measure with the associated insertion, deletion,
    /// and substitution cost.
    pub fn new(insert_cost: usize, delete_cost: usize, substitute_cost: usize) -> Self {
        use self::CaseInsensitiveLevenshteinOp::*;

        CaseInsensitiveLevenshtein {
            ops: [
                Insert(insert_cost),
                Delete(delete_cost),
                Match,
                Substitute(substitute_cost),
            ],
        }
    }
}

impl Measure<char> for CaseInsensitiveLevenshtein {
    type Operation = CaseInsensitiveLevenshteinOp;

    fn operations(&self) -> &[Self::Operation] {
        &self.ops
    }
}

/// Case-insensitive Levenshtein operation with associated cost.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CaseInsensitiveLevenshteinOp {
    Insert(usize),
    Delete(usize),
    Match,
    Substitute(usize),
}

impl Operation<char> for CaseInsensitiveLevenshteinOp {
    fn backtrack(
        &self,
        seq_pair: &SeqPair<char>,
        source_idx: usize,
        target_idx: usize,
    ) -> Option<(usize, usize)> {
        use self::CaseInsensitiveLevenshteinOp::*;

        match *self {
            Delete(cost) => archetype::Delete(cost).backtrack(seq_pair, source_idx, target_idx),
            Insert(cost) => archetype::Insert(cost).backtrack(seq_pair, source_idx, target_idx),
            Match => archetype::Match.backtrack(seq_pair, source_idx, target_idx),
            Substitute(cost) => {
                archetype::Substitute(cost).backtrack(seq_pair, source_idx, target_idx)
            }
        }
    }

    fn cost(
        &self,
        seq_pair: &SeqPair<char>,
        cost_matrix: &[Vec<usize>],
        source_idx: usize,
        target_idx: usize,
    ) -> Option<usize> {
        use self::CaseInsensitiveLevenshteinOp::*;

        let (from_source_idx, from_target_idx) =
            self.backtrack(seq_pair, source_idx, target_idx)?;
        let orig_cost = cost_matrix[from_source_idx][from_target_idx];

        match *self {
            Delete(cost) => {
                archetype::Delete(cost).cost(seq_pair, cost_matrix, source_idx, target_idx)
            }
            Insert(cost) => {
                archetype::Insert(cost).cost(seq_pair, cost_matrix, source_idx, target_idx)
            }
            Match => {
                if iter::once(seq_pair.source[from_source_idx])
                    .default_caseless_match(iter::once(seq_pair.target[from_target_idx]))
                {
                    Some(orig_cost)
                } else {
                    None
                }
            }
            Substitute(cost) => {
                archetype::Substitute(cost).cost(seq_pair, cost_matrix, source_idx, target_idx)
            }
        }
    }
}

/// This function restores uppercase characters in lowercased lemmas from
/// the corresponding forms. This task is actually more complex than it
/// may seem initially due to the properties of Unicode. In particular:
///
/// * Many characters have code points in Unicode, but can also be formed
///   using composed codepoints (e.g. characters with diacritics such as
///   ë). This function applies Normalization Form C to ensure the equivalent
///   representation of characters in the two strings.
/// * Uppercasing or lowercasing a character that is a single code point may
///   result in multiple codepoints. In particular, 'ẞ' (upercased sz) can be
///   lowercased to 'ß' (simple case folding) or 'ss' (full case fulding).
///   This is partially handled --- individual codepoints are compared using
///   Unicode caseless matching. However, if a character is 1 codepoint in
///   the form and >1 codepoint in the lemma (e.g. ẞ vs. ss) or vice versa,
///   the characters will not be matched.
pub(crate) fn restore_named_entity_case<S1, S2>(form: S1, lemma: S2) -> String
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    // Get code points after NFC normalization.
    let form_chars: Vec<char> = form.as_ref().nfc().collect();
    let mut lemma_chars: Vec<char> = lemma.as_ref().nfc().collect();

    // Align the strings using case-insensitive Levenshtein distance.
    let levenshtein = CaseInsensitiveLevenshtein::new(1, 1, 1);
    let script = levenshtein.align(&form_chars, &lemma_chars).edit_script();

    // Copy over aligned characters from the form to the lemma.
    for op in script {
        if let CaseInsensitiveLevenshteinOp::Match = op.operation() {
            lemma_chars[op.target_idx()] = form_chars[op.source_idx()];
        }
    }

    String::from_iter(lemma_chars)
}
