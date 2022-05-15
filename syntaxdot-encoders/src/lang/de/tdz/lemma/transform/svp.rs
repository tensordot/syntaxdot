use std::cmp::Ordering;
use std::collections::VecDeque;

use fst::Set;

use crate::lang::de::tdz::lemma::automaton::Prefixes;
use crate::lang::de::tdz::lemma::constants::*;

/// Candidate list of prefixes and the corresponding stripped form.
struct PrefixesCandidate<'a> {
    stripped_form: &'a str,
    prefixes: Vec<String>,
}

/// Look for all matches of (prefix)* in the given form. Ideally,
/// we'd construct a Kleene star automaton of the prefix automaton.
/// Unfortunately, this functionality is not (yet) provided by the
/// fst crate. Instead, we repeatedly search prefixes in the set.
fn prefix_star<'a, D>(prefix_set: &Set<D>, s: &'a str) -> Vec<PrefixesCandidate<'a>>
where
    D: AsRef<[u8]>,
{
    let mut result = Vec::new();

    let mut q = VecDeque::new();
    q.push_back(PrefixesCandidate {
        stripped_form: s,
        prefixes: Vec::new(),
    });

    while let Some(PrefixesCandidate {
        stripped_form,
        prefixes,
    }) = q.pop_front()
    {
        result.push(PrefixesCandidate {
            stripped_form,
            prefixes: prefixes.clone(),
        });

        for prefix in prefix_set.prefixes(stripped_form) {
            let mut prefixes = prefixes.clone();
            let prefix_len = prefix.len();
            prefixes.push(prefix.to_owned());
            q.push_back(PrefixesCandidate {
                stripped_form: &stripped_form[prefix_len..],
                prefixes,
            });
        }
    }

    result
}

pub fn longest_prefixes<D, F, L, T>(prefix_set: &Set<D>, form: F, lemma: L, tag: T) -> Vec<String>
where
    D: AsRef<[u8]>,
    F: AsRef<str>,
    L: AsRef<str>,
    T: AsRef<str>,
{
    let lemma = lemma.as_ref();
    let form = form.as_ref();
    let tag = tag.as_ref();

    let all_prefixes = prefix_star(prefix_set, form);

    FilterPrefixes {
        inner: all_prefixes.into_iter(),
        lemma,
        tag,
    }
    .max_by(|l, r| {
        match l.stripped_form.len().cmp(&r.stripped_form.len()) {
            Ordering::Less => return Ordering::Greater,
            Ordering::Greater => return Ordering::Less,
            Ordering::Equal => (),
        }

        l.prefixes.len().cmp(&r.prefixes.len()).reverse()
    })
    .map(|t| t.prefixes)
    .unwrap_or_else(Vec::new)
}

fn is_verb<S>(verb: S) -> bool
where
    S: AsRef<str>,
{
    // A separable verb with a length shorter than 3 is unlikely.
    verb.as_ref().len() > 2
}

struct FilterPrefixes<'a, I>
where
    I: Iterator<Item = PrefixesCandidate<'a>>,
{
    lemma: &'a str,
    tag: &'a str,
    inner: I,
}

impl<'a, I> Iterator for FilterPrefixes<'a, I>
where
    I: Iterator<Item = PrefixesCandidate<'a>>,
{
    type Item = PrefixesCandidate<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(candidate) = self.inner.next() {
            if candidate.prefixes.is_empty() {
                return Some(candidate);
            }

            // I don't like the to_owned() here, but as of 1.14-nightly, the
            // borrows checker is not happy about moving candidate otherwise.
            let last_prefix = candidate.prefixes.last().unwrap().to_owned();

            // Avoid e.g. 'dazu' as a valid prefix for a zu-infinitive.
            if self.tag == ZU_INFINITIVE_VERB
                && last_prefix.ends_with("zu")
                && !candidate.stripped_form.starts_with("zu")
            {
                continue;
            }

            // 1. Do not start stripping parts of the lemma
            // 2. Prefix should not end with lemma. E.g.:
            //    abgefangen fangen -> ab#fangen, not: ab#gefangen#fangen
            if candidate.prefixes.iter().any(|p| self.lemma.starts_with(p))
                || last_prefix.ends_with(&self.lemma)
                || !is_verb(candidate.stripped_form)
            {
                continue;
            }

            return Some(candidate);
        }

        None
    }
}
