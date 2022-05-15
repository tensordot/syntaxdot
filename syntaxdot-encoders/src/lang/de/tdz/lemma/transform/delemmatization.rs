//! Delemmatization transformations.
//!
//! This module provides transformations that converts TüBa-D/Z-style lemmas
//! to `regular' lemmas.

use super::{DependencyGraph, Transform};
use crate::lang::de::tdz::lemma::constants::*;

/// Remove alternative lemma analyses.
///
/// TüBa-D/Z sometimes provides multiple lemma analyses for a form. This
/// transformation removes all but the first analysis.
pub struct RemoveAlternatives;

impl Transform for RemoveAlternatives {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let mut lemma = token.lemma();

        if token.xpos().starts_with(PUNCTUATION_PREFIX)
            || token.xpos() == NON_WORD_TAG
            || token.xpos() == FOREIGN_WORD_TAG
        {
            return lemma.to_owned();
        }

        if let Some(idx) = lemma.find('|') {
            lemma = &lemma[..idx];
        }

        lemma.to_owned()
    }
}

/// Replace reflexive tag.
///
/// Reflexives use the special *#refl* lemma in TüBa-D/Z. This transformation
/// replaces this pseudo-lemma by the lowercased form.
pub struct RemoveReflexiveTag;

impl Transform for RemoveReflexiveTag {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();

        if token.xpos() == REFLEXIVE_PERSONAL_PRONOUN_TAG {
            return token.form().to_lowercase();
        }

        lemma.to_owned()
    }
}

/// Remove separable prefixes from verbs.
///
/// TüBa-D/Z marks separable verb prefixes in the verb lemma. E.g. *ab#zeichnen*,
/// where *ab* is the separable prefix. This transformation handles removes
/// separable prefixes from verbs. For example *ab#zeichnen* is transformed to
/// *zeichnen*.
pub struct RemoveSepVerbPrefix;

impl Transform for RemoveSepVerbPrefix {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let mut lemma = token.lemma();

        if is_verb(token.xpos()) {
            if let Some(idx) = lemma.rfind('#') {
                lemma = &lemma[idx + 1..];
            }
        }

        lemma.to_owned()
    }
}

/// Remove truncation markers.
///
/// TüBa-D/Z uses special marking for truncations. For example, *Bau-* in
///
/// *Bau- und Verkehrsplanungen*
///
/// is lemmatized as *Bauplanung%n*, recovering the full lemma and adding
/// a simplified part of speech tag of the word (since the form is tagged
/// as *TRUNC*).
///
/// This transformation replaces the TüBa-D/Z lemma by the word form, such
/// as *Bau-* in this example. If the simplified part of speech tag is not
/// *n*, the lemma is also lowercased.
pub struct RemoveTruncMarker;

impl Transform for RemoveTruncMarker {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();

        if token.xpos() != TRUNCATED_TAG {
            return lemma.to_owned();
        }

        if token.upos() == "NOUN" {
            token.form().to_owned()
        } else {
            token.form().to_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::de::tdz::lemma::transform::test_helpers::run_test_cases;

    use super::{RemoveSepVerbPrefix, RemoveTruncMarker};

    #[test]
    pub fn remove_sep_verb_prefix() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/remove-sep-verb-prefix.test",
            RemoveSepVerbPrefix,
        );
    }

    #[test]
    pub fn remove_trunc_marker() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/remove-trunc-marker.test",
            RemoveTruncMarker,
        );
    }
}
