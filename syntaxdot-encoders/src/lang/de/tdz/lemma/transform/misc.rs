//! Miscellaneous transformations.
//!
//! This module provides transformations that can be used for both
//! lemmatization and delemmatization.

use std::collections::{HashMap, HashSet};

use fst::Set;
use lazy_static::lazy_static;
use maplit::{hashmap, hashset};

use super::{DependencyGraph, Transform};
use crate::lang::de::tdz::lemma::automaton::LongestPrefix;
use crate::lang::de::tdz::lemma::constants::*;

/// Simplify article and relative pronoun lemmas.
///
/// This transformation simplifies lemmas of articles and relative pronouns
/// to *d* for definite and *e* for indefinite. For example:
///
/// * *den* -> *d*
/// * *einem* -> *e*
/// * *dessen* -> *d*
pub struct SimplifyArticleLemma;

impl Transform for SimplifyArticleLemma {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();
        let form = token.form();
        let tag = token.xpos();

        if tag == ARTICLE_TAG || tag == SUBST_REL_PRONOUN || tag == ATTR_REL_PRONOUN {
            if form.to_lowercase().starts_with('d') {
                return String::from("d");
            } else if form.to_lowercase().starts_with('e') {
                return String::from("e");
            }
        }

        lemma.to_owned()
    }
}

lazy_static! {
    static ref PIAT_PREFIXES: Set<Vec<u8>> = Set::from_iter(vec![
        "einig",
        "etlich",
        "irgendein",
        "irgendwelch",
        "jedwed",
        "kein",
        "manch",
        "wenig",
    ])
    .unwrap();
}

/// Simplify attributing indefinite pronouns without determiner (PIAT)
///
/// Simplifies lemmas of this class to some baseform (preliminary) based on matching
/// lowercased prefixes of the forms. The rules are applied in the given order
///
///  "keinerlei" -> "keinerlei"
///  "einig*" -> "einig"
///  "etlich*" -> "etlich"
///  "irgendein*" -> "irgendein"
///  "irgendwelch*" -> "irgendwelch"
///  "jedwed*" -> "jedwed"
///  "kein*" -> "kein"
///  "manch*" -> "manch"
///  "wenig*" -> "wenig"
///
///

pub struct SimplifyPIAT;
impl Transform for SimplifyPIAT {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();
        let form = token.form();
        let tag = token.xpos();

        if tag != ATTRIBUTING_INDEF_PRONOUN_WITHOUT_DET {
            return lemma.to_owned();
        }

        let form = form.to_lowercase();

        if form == "keinerlei" {
            return lemma.to_owned();
        }

        if let Some(prefix) = PIAT_PREFIXES.longest_prefix(&form) {
            return prefix.to_owned();
        }

        lemma.to_owned()
    }
}

lazy_static! {
    static ref PIDAT_LONG_PREFIXES: Set<Vec<u8>> =
        Set::from_iter(vec!["allermeisten", "jedwed", "wenigst"]).unwrap();
    static ref PIDAT_PREFIXES: Set<Vec<u8>> = Set::from_iter(vec![
        "all",
        "ebensolch",
        "ebensoviel",
        "jed",
        "jeglich",
        "meist",
        "solch",
        "soviel",
        "viel",
        "wenig",
        "zuviel",
    ])
    .unwrap();
}

/// Simplify attributing indefinite pronouns with determiner (PIDAT)
///
/// Simplifies lemmas of this class to some baseform (preliminary) based on matching
/// lowercased prefixes of the forms. The rules are applied in the given order
///
///  "allermeisten*" -> "allermeisten"
///  "jedwed*" -> "jedwed"
///  "wenigst*" -> "wenigst"
///  "all*" -> "all"
///  "jede*" -> "jed"
///  "jeglich*" -> "jeglich"
///  "solch*" -> "solch"
///  "ebensolch*" -> "ebensolch"
///  "meist*" -> "meist"
///  "wenigst*" -> "wenigst"
///  "wenig*" -> "wenig"
///  "viele*" -> "viele"
///  "zuviel*" -> "zuviele"
///  "soviel*" -> "soviele"
///  "ebensoviel*" -> "ebensoviele"
///

pub struct SimplifyPIDAT;
impl Transform for SimplifyPIDAT {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();
        let form = token.form();
        let tag = token.xpos();

        if tag != ATTRIBUTING_INDEF_PRONOUN_WITH_DET {
            return lemma.to_owned();
        }

        let form = form.to_lowercase();

        if let Some(prefix) = PIDAT_LONG_PREFIXES.longest_prefix(&form) {
            return prefix.to_owned();
        }

        if let Some(prefix) = PIDAT_PREFIXES.longest_prefix(&form) {
            return prefix.to_owned();
        }

        lemma.to_owned()
    }
}

lazy_static! {
    static ref PIS_LONG_PREFIXES: Set<Vec<u8>> = Set::from_iter(vec![
        "alledem",
        "allerhand",
        "allerlei",
        "allermeisten",
        "einig",
        "einzeln",
        "einzig",
        "jederman",
        "wenigst",
    ])
    .unwrap();
    static ref PIS_PREFIXES: Set<Vec<u8>> = Set::from_iter(vec![
        "alle",
        "ander",
        "beid",
        "ein",
        "erster",
        "etlich",
        "etwas",
        "irgendein",
        "jed",
        "kein",
        "letzter",
        "manch",
        "meist",
        "solch",
        "soviel",
        "viel",
        "wenig",
        "zuviel",
    ])
    .unwrap();
}

/// Simplify attributing indefinite pronouns without determiner (PIAT)
///
/// Simplifies lemmas of this class to some baseform (preliminary) based on matching
/// lowercased prefixes of the forms. The rules are applied in the given order
///
///

pub struct SimplifyPIS;
impl Transform for SimplifyPIS {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let lemma = token.lemma();
        let form = token.form();
        let tag = token.xpos();

        if tag != SUBSTITUTING_INDEF_PRONOUN {
            return lemma.to_owned();
        }

        let form = form.to_lowercase();

        if form.starts_with("andr") {
            return "ander".to_owned();
        }

        if let Some(prefix) = PIS_LONG_PREFIXES.longest_prefix(&form) {
            return prefix.to_owned();
        }

        if let Some(prefix) = PIS_PREFIXES.longest_prefix(&form) {
            return prefix.to_owned();
        }

        lemma.to_owned()
    }
}

lazy_static! {
    static ref PRONOUN_SIMPLIFICATIONS: HashMap<&'static str, HashSet<&'static str>> = hashmap! {
        "ich" => hashset!{"ich", "mich", "mir", "meiner"},
        "du" => hashset!{"du", "dir", "dich", "deiner"},
        "er" => hashset!{"er", "ihn", "ihm", "seiner"},
        "sie" => hashset!{"sie", "ihr", "ihnen", "ihrer"},
        "es" => hashset!{"es", "'s"},
        "wir" => hashset!{"wir", "uns", "unser"},
        "ihr" => hashset!{"euch"} // "ihr"
    };

    static ref PRONOUN_SIMPLIFICATIONS_LOOKUP: HashMap<String, String> =
        inside_out(&PRONOUN_SIMPLIFICATIONS);
}

fn inside_out(map: &HashMap<&'static str, HashSet<&'static str>>) -> HashMap<String, String> {
    let mut new_map = HashMap::new();

    for (&k, values) in map.iter() {
        for &value in values {
            new_map.insert(value.to_owned(), k.to_owned());
        }
    }

    new_map
}

/// Simplify personal pronouns.
///
/// This transformation simplifies personal pronouns using a simple lookup
/// of the lowercased word form. Pronouns are simplified with the following
/// rules (provided by Kathrin Beck):
///
/// Lowercased forms         | Lemma
/// -------------------------|------
/// *ich, mich, mir, meiner* | *ich*
/// *du, dir, dich, deiner*  | *du*
/// *er, ihn, ihm, seiner*   | *er*
/// *sie, ihr, ihnen, ihrer* | *sie*
/// *es, 's*                 | *es*
/// *wir, uns, unser*        | *wir*
/// *ihr, euch*              | *ihr*
///
/// In the case of the ambigious *ihr*, the lemma *sie* is always used.
pub struct SimplifyPersonalPronounLemma;

impl Transform for SimplifyPersonalPronounLemma {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let tag = token.xpos();
        let lemma = token.lemma();

        if tag != PERSONAL_PRONOUN_TAG {
            return lemma.to_owned();
        }

        let form = token.form().to_lowercase();
        if let Some(simplified_lemma) = PRONOUN_SIMPLIFICATIONS_LOOKUP.get(&form) {
            simplified_lemma.to_owned()
        } else {
            lemma.to_owned()
        }
    }
}

lazy_static! {
    static ref ATTR_POSS_PRONOUN_PREFIXES: Set<Vec<u8>> =
        Set::from_iter(vec!["dein", "euer", "eure", "ihr", "mein", "sein", "unser"]).unwrap();
    static ref SUBST_POSS_PRONOUN_PREFIXES: Set<Vec<u8>> =
        Set::from_iter(vec!["dein", "ihr", "mein", "sein", "unser", "unsrig"]).unwrap();
}

/// Simplify possesive pronoun lemmas.
///
/// This transformation simplifies pronoun lemmas to lemmas without
/// gender-specific suffixes. For example:
///
/// * *deinen* -> *dein*
/// * *deiner* -> *dein*
pub struct SimplifyPossesivePronounLemma;

impl Transform for SimplifyPossesivePronounLemma {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String {
        let token = graph.token(node);
        let tag = token.xpos();
        let form = token.form();
        let lemma = token.lemma();

        if tag != ATTRIBUTIVE_POSSESIVE_PRONOUN_TAG && tag != SUBST_POSSESIVE_PRONOUN_TAG {
            return lemma.to_owned();
        }

        let form = form.to_lowercase();
        let prefix = if tag == SUBST_POSSESIVE_PRONOUN_TAG {
            SUBST_POSS_PRONOUN_PREFIXES.longest_prefix(&form)
        } else {
            ATTR_POSS_PRONOUN_PREFIXES.longest_prefix(&form)
        };

        if let Some(mut prefix) = prefix {
            if prefix == "eure" {
                prefix = "euer";
            }

            return prefix.to_owned();
        }

        lemma.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::de::tdz::lemma::transform::test_helpers::run_test_cases;

    use super::{
        SimplifyArticleLemma, SimplifyPIAT, SimplifyPIDAT, SimplifyPIS,
        SimplifyPersonalPronounLemma, SimplifyPossesivePronounLemma,
    };

    #[test]
    pub fn simplify_pidat_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-pidat-lemma.test",
            SimplifyPIDAT,
        );
    }

    #[test]
    pub fn simplify_article_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-article-lemma.test",
            SimplifyArticleLemma,
        );
    }

    #[test]
    pub fn simplify_piat_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-piat-lemma.test",
            SimplifyPIAT,
        );
    }

    #[test]
    pub fn simplify_pis_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-pis-lemma.test",
            SimplifyPIS,
        );
    }

    #[test]
    pub fn simplify_possesive_pronoun_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-possesive-pronoun-lemma.test",
            SimplifyPossesivePronounLemma,
        );
    }

    #[test]
    pub fn simplify_personal_pronoun_lemma() {
        run_test_cases(
            "testdata/lang/de/tdz/lemma/simplify-personal-pronoun.test",
            SimplifyPersonalPronounLemma,
        );
    }
}
