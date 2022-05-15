use std::collections::HashSet;

use lazy_static::lazy_static;
use maplit::hashset;

pub(crate) static REFLEXIVE_PERSONAL_PRONOUN_LEMMA: &str = "#refl";

pub(crate) static SEPARABLE_PARTICLE_POS: &str = "PTKVZ";

pub(crate) static PUNCTUATION_PREFIX: &str = "$";

pub(crate) static ARTICLE_TAG: &str = "ART";
pub(crate) static ATTRIBUTIVE_POSSESIVE_PRONOUN_TAG: &str = "PPOSAT";
pub(crate) static SUBST_POSSESIVE_PRONOUN_TAG: &str = "PPOSS";
pub(crate) static FOREIGN_WORD_TAG: &str = "FM";
pub(crate) static NAMED_ENTITY_TAG: &str = "NE";
pub(crate) static NON_WORD_TAG: &str = "XY";
pub(crate) static NOUN_TAG: &str = "NN";
pub(crate) static PERSONAL_PRONOUN_TAG: &str = "PPER";
pub(crate) static REFLEXIVE_PERSONAL_PRONOUN_TAG: &str = "PRF";
pub(crate) static SUBST_REL_PRONOUN: &str = "PRELS";
pub(crate) static ATTR_REL_PRONOUN: &str = "PRELAT";
pub(crate) static TRUNCATED_TAG: &str = "TRUNC";
pub(crate) static ZU_INFINITIVE_VERB: &str = "VVIZU";

pub(crate) static SUBSTITUTING_INDEF_PRONOUN: &str = "PIS";
pub(crate) static ATTRIBUTING_INDEF_PRONOUN_WITHOUT_DET: &str = "PIAT";
pub(crate) static ATTRIBUTING_INDEF_PRONOUN_WITH_DET: &str = "PIDAT";

lazy_static! {
    pub(crate) static ref LEMMA_IS_FORM_TAGS: HashSet<&'static str> = hashset! {
        "$,",
        "$.",
        "$(",
        "ADV",
        "APPR",
        "APPO",
        "APZR",
        "ITJ",
        "KOUI",
        "KOUS",
        "KON",
        "KOKOM",
        "ADJD",
        "CARD",
        "PTKZU",
        "PTKA",
        "PTKNEG",
    };
    pub(crate) static ref LEMMA_IS_FORM_PRESERVE_CASE_TAGS: HashSet<&'static str> = hashset! {
        FOREIGN_WORD_TAG,
    };
}

pub(crate) fn is_verb<S>(tag: S) -> bool
where
    S: AsRef<str>,
{
    tag.as_ref().starts_with('V')
}

pub(crate) fn is_separable_verb<S>(tag: S) -> bool
where
    S: AsRef<str>,
{
    let tag = tag.as_ref();
    tag == "VVFIN" || tag == "VVPP" || tag == "VVIMP" || tag == "VMFIN" || tag == "VAFIN"
}
