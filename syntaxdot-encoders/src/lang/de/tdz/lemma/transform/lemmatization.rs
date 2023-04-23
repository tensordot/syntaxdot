//! Lemmatization transformations.
//!
//! This module provides transformations that converts lemmas to TüBa-D/Z-style
//! lemmas.

use std::collections::HashMap;
use std::io::{BufRead, Cursor};

use fst::{Set, SetBuilder};

use super::named_entity::restore_named_entity_case;
use super::svp::longest_prefixes;
use super::{DependencyGraph, Transform};
use crate::lang::de::tdz::lemma::constants::*;
use crate::lang::de::tdz::lemma::error::LemmatizationError;

/// Set the lemma of reflexive personal pronouns (PRF) to `#refl`.
pub struct AddReflexiveTag;

impl Transform for AddReflexiveTag {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();

        if token.xpos() == REFLEXIVE_PERSONAL_PRONOUN_TAG {
            REFLEXIVE_PERSONAL_PRONOUN_LEMMA.to_owned()
        } else {
            lemma.to_owned()
        }
    }
}

/// Add separable verb prefixes to verbs.
///
/// TüBa-D/Z marks separable verb prefixes in the verb lemma. E.g. *ab#zeichnen*,
/// where *ab* is the separable prefix. This transformation handles cases where
/// the prefix is separated from the verb. For example, in the sentence
///
/// *Diese änderungen zeichnen sich bereits ab .*
///
/// The transformation rule will lemmatize *zeichnen* to *ab#zeichnen*. The
/// separable particle of a verb is found using dependency structure. In some
/// limited cases, it will also handle verbs with multiple `competing' separable
/// prefixes. For example, *nimmt* in
///
/// *[...] nimmt eher zu als ab*
///
/// is lemmatized as *zu#nehmen|ab#nehmen*.
pub struct AddSeparatedVerbPrefix {
    multiple_prefixes: bool,
}

impl AddSeparatedVerbPrefix {
    pub fn new(multiple_prefixes: bool) -> Self {
        AddSeparatedVerbPrefix { multiple_prefixes }
    }
}

impl Transform for AddSeparatedVerbPrefix {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();

        if !is_separable_verb(token.xpos()) {
            return lemma.to_owned();
        }

        let mut lemma = lemma.to_owned();

        // Find all nodes that are attached with the separable verb dependency
        // relation.
        //
        // Fixme: check AVZ/KON relation as well?
        // Fixme: what about particles linked KON?
        let mut prefix_iter = graph
            .dependents(node)
            .filter(|(dependent, _)| graph.token(*dependent).xpos() == SEPARABLE_PARTICLE_POS);

        if self.multiple_prefixes {
            let mut lemmas = Vec::new();

            // Fixme: prefixes are not returned in sentence order?
            for (dependant, _) in prefix_iter {
                let prefix = graph.token(dependant);
                lemmas.push(format!("{}#{}", prefix.form().to_lowercase(), lemma));
            }

            if lemmas.is_empty() {
                lemma
            } else {
                lemmas.join("|")
            }
        } else {
            if let Some((dependant, _)) = prefix_iter.next() {
                let prefix = graph.token(dependant);
                lemma.insert_str(0, &format!("{}#", prefix.form().to_lowercase()));
            }

            lemma
        }
    }
}

/// Lemmatize tokens where the form is the lemma.
pub struct FormAsLemma;

impl Transform for FormAsLemma {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);

        // Handle tags for which the lemma is the lowercased form.
        if LEMMA_IS_FORM_TAGS.contains(token.xpos()) {
            token.form().to_lowercase()
        } else if LEMMA_IS_FORM_PRESERVE_CASE_TAGS.contains(token.xpos()) {
            token.form().to_owned()
        } else {
            token.lemma().to_owned()
        }
    }
}

/// Mark separable verb prefixes in verbs.
///
/// TüBa-D/Z marks separable verb prefixes in the verb lemma. E.g. *ab#zeichnen*,
/// where *ab* is the separable prefix. This transformation handles cases where
/// the prefix is **not** separated from the verb. For example, it makes the
/// following transformations:
///
/// 1. *abhing/hängen* -> *abhängen*
/// 2. *dazugefügt/fügen* -> *dazu#fügen*
/// 3. *wiedergutgemacht/machen* -> *wieder#gut#machen*
/// 4. *hinzubewegen/bewegen* -> *hin#bewegen*
///
/// The transformation rule prefers analysis with longer prefixes over shorter
/// prefixes. This leads to the analysis (2) rather than *da#zu#fügen*.
///
/// When a verb contains multiple separable prefixes, this transformation rule
/// attempts to find them, as in (3).
///
/// In 'zu'-infinitives *zu* is removed and not analyzed as being (part of) a
/// separable prefix.
pub struct MarkVerbPrefix {
    prefix_verbs: HashMap<String, String>,
    prefixes: Set<Vec<u8>>,
}

impl MarkVerbPrefix {
    /// Create this transformation. A simple lookup for prefix verbs can be
    /// provided. More crucially, a set of prefixes must be provided to find
    /// prefixes.
    pub fn new() -> Self {
        MarkVerbPrefix::read_verb_prefixes(Cursor::new(include_str!(
            "../../../../../../data/lang/de/tdz/tdz11-separable-prefixes.txt"
        )))
        .expect("Invalid separable verb prefix data")
    }

    #[allow(unused)]
    pub fn set_prefix_verbs(&mut self, prefix_verbs: HashMap<String, String>) {
        self.prefix_verbs = prefix_verbs;
    }
}

impl Default for MarkVerbPrefix {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for MarkVerbPrefix {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();
        let lemma_lc = lemma.to_lowercase();

        if !is_verb(token.xpos()) {
            return lemma.to_owned();
        }

        // There are two cases that we have to handle separately:
        //
        // 1. The lemmatizer did not strip the prefix. In this case, we
        //    perform a lemma lookup. For now, removing prefixes from the
        //    lemma itself seems to be too tricky.
        //
        // 2. The lemmatizer stripped the prefix. The prefix needs to be
        //    inferred from the token's form.

        // Case 1: try a simple lookup for the lemma
        if let Some(sep_lemma) = self.prefix_verbs.get(&lemma_lc) {
            return sep_lemma.clone();
        }

        // Case 2: there are no prefixes in the lemma, try to find prefixes
        // in the form.
        let form_lc = token.form().to_lowercase();
        let mut lemma_parts = longest_prefixes(&self.prefixes, form_lc, &lemma_lc, token.xpos());
        if !lemma_parts.is_empty() {
            lemma_parts.push(lemma_lc);
            return lemma_parts.join("#");
        }

        lemma.to_owned()
    }
}

trait ReadVerbPrefixes {
    fn read_verb_prefixes<R>(r: R) -> Result<MarkVerbPrefix, LemmatizationError>
    where
        R: BufRead;
}

impl ReadVerbPrefixes for MarkVerbPrefix {
    fn read_verb_prefixes<R>(r: R) -> Result<MarkVerbPrefix, LemmatizationError>
    where
        R: BufRead,
    {
        let mut builder = SetBuilder::memory();

        for line in r.lines() {
            let line = line?;

            builder.insert(&line)?;
        }

        let bytes = builder.into_inner()?;
        let prefixes = Set::new(bytes)?;

        Ok(MarkVerbPrefix {
            prefix_verbs: HashMap::new(),
            prefixes,
        })
    }
}

pub struct RestoreCase;

impl Transform for RestoreCase {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);

        if token.xpos() == NOUN_TAG {
            uppercase_first_char(token.lemma())
        } else if token.xpos() == NAMED_ENTITY_TAG {
            restore_named_entity_case(token.form(), token.lemma())
        } else {
            token.lemma().to_owned()
        }
    }
}

fn uppercase_first_char<S>(s: S) -> String
where
    S: AsRef<str>,
{
    // Hold your seats... This is a bit convoluted, because uppercasing a
    // unicode codepoint can result in multiple codepoints. Although this
    // should not hapen in German orthography, we want to be correct here...

    let mut chars = s.as_ref().chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return String::new(),
    };

    first.to_uppercase().chain(chars).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::iter::FromIterator;

    use crate::lang::de::tdz::lemma::transform::test_helpers::run_test_cases;

    use super::{
        uppercase_first_char, AddSeparatedVerbPrefix, FormAsLemma, MarkVerbPrefix, RestoreCase,
    };

    #[test]
    pub fn first_char_is_uppercased() {
        assert_eq!(uppercase_first_char("test"), "Test");
        assert_eq!(uppercase_first_char("Test"), "Test");
        assert_eq!(uppercase_first_char(""), "");
    }

    #[test]
    pub fn add_separated_verb_prefix() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/add-separated-verb-prefix.test",
            AddSeparatedVerbPrefix {
                multiple_prefixes: true,
            },
        );
    }

    #[test]
    pub fn form_as_lemma() {
        run_test_cases("testdata/lang/de/tdz/lemma/form-as-lemma.test", FormAsLemma);
    }

    #[test]
    pub fn mark_verb_prefix() {
        let prefix_verbs = HashMap::from_iter(vec![(
            String::from("abbestellen"),
            String::from("ab#bestellen"),
        )]);

        let mut transform = MarkVerbPrefix::new();
        transform.set_prefix_verbs(prefix_verbs);

        run_test_cases(
            "testdata/lang/de/tdz/lemma/mark-verb-prefix.test",
            transform,
        );
    }

    #[test]
    pub fn restore_case() {
        run_test_cases("testdata/lang/de/tdz/lemma/restore-case.test", RestoreCase);
    }
}
